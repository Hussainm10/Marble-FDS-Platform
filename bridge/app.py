"""FastAPI bridge service: webhook receiver, Marbel/GNN escalation, decision processing.

Canonical behaviors wired into this app:
- Error model (``bridge/errors.py``): uniform ``{code, message, details}`` with
  canonical codes (AGT-401/402/429, FDS-451, DOC-415, FDS-500). Internal cause
  strings are suppressed unless ``DEBUG_DETAILS=true`` in env.
- Idempotency (``bridge/idempotency.py``): write endpoints accept an
  ``Idempotency-Key`` header; same key + same body returns the cached
  response without re-executing the work.
- Structured logging (``structlog``) + correlation ID (``asgi-correlation-id``):
  every log line within a request carries the same correlation_id, and the ID
  is propagated to Marbel/GNN via the ``X-Correlation-ID`` header.
- Metrics (``prometheus-fastapi-instrumentator``): exposes ``/metrics`` with
  request-duration histograms and status-code counters, plus custom counters
  for decisions, escalations, STR/CTR drafts.
- Optional bridge auth (set ``BRIDGE_API_KEY`` env) and webhook HMAC
  verification (set ``WEBHOOK_SECRET``). Unset = off (dev mode).
"""

import hashlib
import hmac
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from .checkmarble_client import CheckmarbleClient
from .compliance import process_compliance
from .config import settings
from .errors import FDSError, register_exception_handlers
from .escalation import escalate
from .idempotency import IdempotencyMiddleware
from .ingest_validators import validate_ingest_payload, ForbiddenFieldError
from .models import (
    CheckmarbleDecision,
    DecideRequest,
    EscalationResult,
    IngestRequest,
    WebhookEvent,
)


# ---------------------------------------------------------------------------
# Structured logging: JSON output + correlation_id injection
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(message)s",
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
)


def _inject_correlation_id(_logger, _method, event_dict):
    cid = correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _inject_correlation_id,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger("bridge")


# ---------------------------------------------------------------------------
# Metrics (custom counters alongside the default instrumentator histograms)
# ---------------------------------------------------------------------------
M_DECISION_TOTAL = Counter(
    "fds_decision_total", "Total /decide calls", ["outcome"]
)
M_ESCALATION_TOTAL = Counter(
    "fds_escalation_total", "Total escalations triggered", ["risk_level"]
)
M_STR_TOTAL = Counter("fds_str_total", "Total STR drafts generated")
M_CTR_TOTAL = Counter("fds_ctr_total", "Total CTR filings generated")
# the platform's compliance hard-lock policy §13 / §M11 — fail-safe instrumentation.
from prometheus_client import Gauge
M_CIRCUIT_STATE = Gauge(
    "marble_fds_circuit_state",
    "Marble FDS circuit breaker state (0=closed, 1=open, 2=half_open)",
)
M_DECIDE_ERRORS = Counter(
    "marble_fds_decide_errors_total",
    "Errors when calling Marble /decide", ["error_type"],
)
M_FALLBACK_ACTIVATIONS = Counter(
    "marble_fds_fallback_activations_total",
    "Number of times the fail-safe fallback was returned (Marble bypassed)",
)


client = CheckmarbleClient()
idempotency = IdempotencyMiddleware()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("bridge_start",
             bridge_api_key_enforced=bool(settings.BRIDGE_API_KEY),
             webhook_hmac_enforced=bool(settings.WEBHOOK_SECRET))
    yield
    await client.close()
    log.info("bridge_stop")


app = FastAPI(
    title="Marble FDS Bridge Service",
    description="Webhook receiver and Marbel/GNN escalation bridge for Checkmarble decisions",
    version="1.0.0",
    lifespan=lifespan,
)

# Correlation ID must be the outermost middleware so every downstream log gets it.
app.add_middleware(CorrelationIdMiddleware)

register_exception_handlers(app)

# /metrics endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ---------------------------------------------------------------------------
# Helpers: auth + webhook HMAC (both opt-in via env)
# ---------------------------------------------------------------------------

# Endpoints that allow anonymous access (health, metrics).
_PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}


def _require_bridge_auth(request: Request) -> None:
    """Raise 401 if BRIDGE_API_KEY is set and the request header doesn't match."""
    if not settings.BRIDGE_API_KEY:
        return
    if request.url.path in _PUBLIC_PATHS:
        return
    presented = request.headers.get("X-Bridge-API-Key", "")
    if not hmac.compare_digest(presented, settings.BRIDGE_API_KEY):
        raise HTTPException(status_code=401, detail="invalid_or_missing_X-Bridge-API-Key")


def _verify_webhook_signature(raw_body: bytes, signature_header: str) -> None:
    """Raise 401 if WEBHOOK_SECRET is set and HMAC-SHA256 doesn't match."""
    if not settings.WEBHOOK_SECRET:
        return
    expected = hmac.new(
        settings.WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    # Accept both "sha256=<hex>" and bare "<hex>".
    candidate = signature_header.removeprefix("sha256=") if signature_header else ""
    if not candidate or not hmac.compare_digest(candidate, expected):
        raise HTTPException(status_code=401, detail="invalid_webhook_signature")


@app.middleware("http")
async def bridge_auth_middleware(request: Request, call_next):
    # Webhook endpoint uses HMAC (handled inside the route); everything else
    # uses the X-Bridge-API-Key header if BRIDGE_API_KEY is set.
    if request.url.path != "/webhooks/checkmarble":
        _require_bridge_auth(request)
    return await call_next(request)


def _sanitize_cause(exc: Exception) -> str:
    """Return a safe error-details string. Full exception only in DEBUG_DETAILS mode."""
    if settings.DEBUG_DETAILS:
        return str(exc)[:500]
    return "internal_service_error"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    checkmarble_ok = await client.health()
    return {
        "status": "healthy",
        "checkmarble_connected": checkmarble_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/webhooks/checkmarble")
async def webhook_receiver(http_request: Request, event: WebhookEvent) -> dict:
    """Receives Checkmarble decision events. Triggers escalation if score >= threshold.

    Idempotency-Key header is honored so repeated webhook deliveries don't
    trigger duplicate escalations.
    """
    raw_body = await http_request.body()
    _verify_webhook_signature(raw_body, http_request.headers.get("X-Webhook-Signature", ""))

    async def do_work():
        log.info("webhook_received", webhook_event=event.event,
                 decision_id=event.decision_id, score=event.score)

        if event.event == "decision.created" and event.score >= settings.MARBEL_ESCALATION_THRESHOLD:
            try:
                decision_data = await client.get_decision(event.decision_id)
                d = decision_data.get("decision", decision_data)
                decision = CheckmarbleDecision(
                    decision_id=event.decision_id,
                    outcome=event.outcome,
                    score=event.score,
                    trigger_object_type=d.get("trigger_object_type", ""),
                    trigger_object_id=d.get("trigger_object_id", ""),
                )
                # Best-effort: Checkmarble's /decisions/{id} sometimes returns the
                # trigger_object inline. If it's there, use it for GNN node-features.
                webhook_trigger = d.get("trigger_object") if isinstance(d, dict) else None
                result = await escalate(decision, trigger_object=webhook_trigger)
                M_ESCALATION_TOTAL.labels(risk_level=result.risk_level.value).inc()
                return {"status": "escalated", "escalation": result.model_dump(mode="json")}
            except Exception as e:
                log.error("webhook_escalation_failed", decision_id=event.decision_id, exc_info=True)
                return {
                    "status": "escalation_failed",
                    "error": _sanitize_cause(e),
                    "decision_id": event.decision_id,
                }

        if event.event == "decision.updated":
            log.info("decision_updated", decision_id=event.decision_id,
                     outcome=event.outcome, reviewed_by=event.reviewed_by)
            return {"status": "acknowledged", "event": event.event}

        log.info("webhook_ignored", webhook_event=event.event)
        return {"status": "received", "event": event.event}

    return await idempotency.process(http_request, do_work)


@app.post("/ingest/{table_name}")
async def ingest_proxy(
    table_name: str,
    request: IngestRequest,
    http_request: Request,
) -> dict:
    # Hard-lock validators (the platform's compliance hard-lock policy §M5 + §M12). Reject card data
    # and MSISDN-as-key BEFORE the payload is forwarded to Marble.
    validate_ingest_payload(table_name, request.object_id, request.data)

    async def do_work():
        try:
            result = await client.ingest(table_name, request.object_id, request.data)
            return {"status": "ingested", "table": table_name, "result": result}
        except Exception as e:
            log.error("ingestion_failed", table=table_name, object_id=request.object_id,
                      exc_info=True)
            raise FDSError(
                message=f"Checkmarble ingestion failed for {table_name}",
                details={"table": table_name, "object_id": request.object_id,
                         "cause": _sanitize_cause(e)},
            )

    return await idempotency.process(http_request, do_work)


@app.post("/decide", response_model=None)
async def decide(request: DecideRequest, http_request: Request) -> dict:
    """Calls Checkmarble decision API, processes response, escalates if needed.

    Total wall-clock latency (decide → escalate → compliance) is written back to
    escalation.latency_ms per spec.md §7.4.
    """
    # Hard-lock validators on the trigger object (the platform's compliance hard-lock policy §M5 + §M12).
    # Reject card data and MSISDN-as-key before any Marble interaction.
    validate_ingest_payload(
        request.trigger_object_type,
        request.trigger_object.get("object_id") if isinstance(request.trigger_object, dict) else None,
        request.trigger_object,
    )

    async def do_work():
        from .circuit_breaker import marble_circuit, CircuitState
        from .operational_events import (
            map_score_to_event, event_to_customer_string,
            is_compliance_caller, build_minimal_response, OperationalEvent,
            ApprovedCustomerString,
        )
        _start = time.perf_counter()

        # ── M11 / §13 circuit-breaker fail-safe ─────────────────────────
        # If the breaker is OPEN, do NOT call Marble. Return a fail-safe
        # response so customer transactions continue under base rules.
        # Per §13.2: any txn at/above LCTR threshold still creates a
        # Compliance queue entry regardless of FDS availability — emitted
        # via structured log with bypass=True so post-hoc review is possible.
        M_CIRCUIT_STATE.set(int(marble_circuit.state))
        if not marble_circuit.can_proceed():
            M_FALLBACK_ACTIVATIONS.inc()
            log.warning(
                "marble_circuit_open_failsafe",
                bypass=True,
                amount=request.trigger_object.get("amount") if isinstance(request.trigger_object, dict) else None,
                scenario_id=request.scenario_id,
            )
            txn_id = ""
            if isinstance(request.trigger_object, dict):
                txn_id = (request.trigger_object.get("transaction_id")
                          or request.trigger_object.get("object_id") or "")
            return {
                "transaction_id": txn_id,
                "operational_event": OperationalEvent.PASS_THROUGH.value,
                "status": ApprovedCustomerString.PROCESSING.value,
                "fail_safe": True,
                "fallback_reason": "marble_unavailable",
            }

        try:
            decision = await client.decide(
                scenario_id=request.scenario_id,
                trigger_object_type=request.trigger_object_type,
                trigger_object=request.trigger_object,
            )
            marble_circuit.record_success()
            M_CIRCUIT_STATE.set(int(marble_circuit.state))
        except Exception as e:
            marble_circuit.record_failure()
            M_DECIDE_ERRORS.labels(error_type=type(e).__name__).inc()
            M_CIRCUIT_STATE.set(int(marble_circuit.state))
            log.error("decision_request_failed", scenario_id=request.scenario_id, exc_info=True)
            raise FDSError(
                message="Checkmarble decision request failed",
                details={"scenario_id": request.scenario_id, "cause": _sanitize_cause(e)},
            )

        M_DECISION_TOTAL.labels(outcome=decision.outcome or "unknown").inc()
        response: dict = {"decision": decision.model_dump()}

        escalation_result: EscalationResult | None = None
        if decision.score >= settings.MARBEL_ESCALATION_THRESHOLD:
            try:
                escalation_result = await escalate(decision, trigger_object=request.trigger_object)
                M_ESCALATION_TOTAL.labels(risk_level=escalation_result.risk_level.value).inc()
            except Exception as e:
                log.error("escalation_failed", exc_info=True)
                response["escalation_error"] = _sanitize_cause(e)

        # process_compliance mutates escalation_result in place (sanctions_match,
        # shariah flags, ctr_required, audit_log_reference, str_suggested).
        try:
            compliance = await process_compliance(
                decision, escalation_result, request.trigger_object,
            )
            response["compliance"] = compliance
            if compliance.get("str_generated"):
                M_STR_TOTAL.inc()
            if compliance.get("ctr_generated"):
                M_CTR_TOTAL.inc()
        except Exception as e:
            log.error("compliance_failed", exc_info=True)
            response["compliance_error"] = _sanitize_cause(e)

        if escalation_result is not None:
            escalation_result.latency_ms = int((time.perf_counter() - _start) * 1000)
            response["escalation"] = escalation_result.model_dump(mode="json")

        # ── the operator operational event remap (the platform's compliance hard-lock policy §4.1, §M1) ──
        # Attach the operational event + neutral customer string on every
        # response, including PASS_THROUGH / SILENT_REVIEW (no escalation).
        # M1 hard-lock: bridge produces the event but performs no automated
        # account action. Enforcement is the operator's back-office maker–checker.
        from .operational_events import (
            map_score_to_event, event_to_customer_string,
            is_compliance_caller, build_minimal_response,
        )
        op_event = map_score_to_event(decision.score)
        response["operational_event"] = op_event.value
        response["customer_status_string"] = event_to_customer_string(op_event).value

        # ── Response visibility gate (the platform's compliance hard-lock policy §M4 / §11.2) ──
        # In "minimal" mode, only Compliance roles see the full FDS payload.
        # All other callers (app/agent/merchant/internal-non-compliance)
        # get the customer-facing minimal response: txn_id + neutral status.
        if settings.BRIDGE_RESPONSE_MODE == "minimal" and not is_compliance_caller(
            http_request.headers.get("X-Compliance-Role")
        ):
            txn_id = ""
            if isinstance(request.trigger_object, dict):
                txn_id = (request.trigger_object.get("transaction_id")
                          or request.trigger_object.get("object_id") or "")
            return build_minimal_response(
                transaction_id=txn_id,
                operational_event=op_event,
                customer_status_string=event_to_customer_string(op_event),
            )

        return response

    return await idempotency.process(http_request, do_work)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
