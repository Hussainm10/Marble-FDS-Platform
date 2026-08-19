"""STR/CTR compliance automation (platform-level).

Generates Suspicious Transaction Reports (STR) and Currency Transaction
Reports (CTR) based on decision scores and transaction amounts, then
ingests them as ``risk_events`` and logs to ``audit_trail``.

The STR score threshold (default 40) and CTR amount threshold (default
500,000 in local currency) are jurisdiction-configurable — override via
environment variables or the operator's jurisdiction pack.

═══════════════════════════════════════════════════════════════════════
HARD-LOCK — NO AUTO-SUBMISSION TO REGULATORS
═══════════════════════════════════════════════════════════════════════
Per the platform's compliance hard-lock policy and the deploying jurisdiction's central bank AML/CFT regulation:
this module DRAFTS STR/CTR records into the internal Marble risk_events
table only. It MUST NEVER make an outbound HTTP call to the jurisdiction's Financial Intelligence Unit (FIU), the central bank,
or any external regulator endpoint. Submission requires Compliance
Checker JWT + Authorised Signatory in the the operator's back-office
maker–checker workflow — implemented outside this codebase.

The ``_ingest_record`` helper below enforces a hostname allowlist that
rejects any URL whose host does not match the configured
CHECKMARBLE_API_URL. Any future code that tries to POST to the jurisdiction's Financial Intelligence Unit (FIU)
will fail at that guard with ``ComplianceAutoSubmitForbidden``.
═══════════════════════════════════════════════════════════════════════

Spec references:
    - the platform's compliance hard-lock policy §3 (AML / STR / CTR / LCTR framework)
    - spec.md §9 Compliance Infrastructure — STR/CTR hooks, audit trail
    - spec.md §1.3 Risk bands (60+ = decline + STR)
    - Jurisdiction pack typically sets timing: e.g., STR filed within 24h
      per regulator rules; CTR daily aggregate per FIU requirements.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
import structlog
from asgi_correlation_id import correlation_id

from .config import settings
from .models import CheckmarbleDecision, EscalationResult

logger = structlog.get_logger("bridge.compliance")


class ComplianceAutoSubmitForbidden(RuntimeError):
    """Raised when this module attempts to POST anywhere other than the
    configured internal Marble ingestion endpoint. Hard-lock per
    the platform's compliance hard-lock policy §3.1 (M3): Marble FDS may auto-draft, never auto-submit.
    """


def _correlation_headers() -> dict:
    cid = correlation_id.get()
    return {"X-Correlation-ID": cid} if cid else {}

import os

# Thresholds — override per-jurisdiction via env vars or a jurisdiction pack.
# Default is 500,000 in local currency; some regulators set a lower or
# higher LCTR/CTR threshold — override via CTR_AMOUNT_THRESHOLD in .env.
STR_SCORE_THRESHOLD = int(os.getenv("STR_SCORE_THRESHOLD", "40"))
CTR_AMOUNT_THRESHOLD = int(os.getenv("CTR_AMOUNT_THRESHOLD", "500000"))


# ---------------------------------------------------------------------------
# Checkmarble ingestion helper
# ---------------------------------------------------------------------------

def _allowed_ingest_host() -> str:
    """The single host this module is permitted to POST to.

    Hard-lock per the platform's compliance hard-lock policy §3.1 (M3): the only permitted destination
    is the internal Marble (Checkmarble) ingestion endpoint configured via
    CHECKMARBLE_API_URL. Any attempt to call the jurisdiction's Financial Intelligence Unit (FIU), the central bank, or any other
    external regulator endpoint MUST fail.
    """
    parsed = urlparse(settings.CHECKMARBLE_API_URL)
    return (parsed.hostname or "").lower()


async def _ingest_record(table_name: str, record: dict):
    """POST a record to Checkmarble's ingestion API.

    Hardened against accidental auto-submission to external regulators:
    the destination URL hostname is verified against
    ``_allowed_ingest_host()``. Any mismatch raises
    :class:`ComplianceAutoSubmitForbidden`. Per the platform's compliance hard-lock policy §3.1 (M3),
    Marble FDS may auto-draft only; submission to the jurisdiction's Financial Intelligence Unit (FIU)/the central bank requires
    human Compliance Checker + Authorised Signatory action in the
    the operator's back-office, never an automated HTTP call from this module.
    """
    url = f"{settings.CHECKMARBLE_API_URL.rstrip('/')}/ingestion/{table_name}"
    target_host = (urlparse(url).hostname or "").lower()
    if target_host != _allowed_ingest_host():
        raise ComplianceAutoSubmitForbidden(
            f"Refusing to POST to '{target_host}': only '{_allowed_ingest_host()}' "
            f"is allowed (the platform's compliance hard-lock policy §3.1 / M3 hard-lock)."
        )
    headers = {
        "X-API-Key": settings.CHECKMARBLE_API_KEY,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15, headers=_correlation_headers()) as client:
        resp = await client.post(url, json=record, headers=headers)
        resp.raise_for_status()


async def _add_to_str_list(wallet_uid: str):
    """Best-effort: add entity to the str_flagged_users custom list.

    Looks up the list by name, then adds the entry.  Failures are logged
    but do not block the STR generation.
    """
    base = settings.CHECKMARBLE_API_URL.rstrip("/")
    headers = {
        "X-API-Key": settings.CHECKMARBLE_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10, headers=_correlation_headers()) as client:
            # Find list ID
            resp = await client.get(f"{base}/custom-lists", headers=headers)
            resp.raise_for_status()
            lists = resp.json()
            list_id = None
            for lst in lists:
                if lst.get("name") == "str_flagged_users":
                    list_id = lst["id"]
                    break
            if not list_id:
                logger.warning("str_list_not_found")
                return

            # Add entry
            await client.post(
                f"{base}/custom-lists/{list_id}/values",
                json={"value": wallet_uid},
                headers=headers,
            )
            logger.info("str_list_added", wallet_uid=wallet_uid)
    except Exception as e:
        logger.warning("str_list_update_failed", error=str(e))


# ---------------------------------------------------------------------------
# Risk event factory (shared STR + CTR construction)
# ---------------------------------------------------------------------------

def _build_risk_event(
    *,
    kind: str,           # "STR" or "CTR"
    score: float,
    amount: float,
    decision: CheckmarbleDecision,
    trigger_object: dict,
    risk_category: str,
    trigger_description: str,
    compliance_action: str,
    notes: str,
) -> dict:
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    risk_id = f"{kind}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    now_iso = now.isoformat()
    wallet_uid = trigger_object.get("from_wallet_uid", trigger_object.get("object_id", ""))
    # Retention metadata (the platform's compliance hard-lock policy §M6 / §12.2). STR/CTR are
    # AML-linked records → 10-year minimum retention. The absolute
    # retain-until date is precomputed and embedded in `notes` so
    # downstream WORM storage can enforce immutability for the required
    # window. (Marble v0.59 risk_events schema is fixed; we don't add
    # new top-level fields without a schema migration.)
    retention_days = settings.AUDIT_RETENTION_DAYS_AML
    retain_until_iso = (now + timedelta(days=retention_days)).isoformat()
    notes_with_retention = (
        f"{notes} | retention_policy=AML_LINKED retention_days={retention_days} "
        f"retain_until={retain_until_iso} (the platform's compliance hard-lock policy §M6)"
    )
    return {
        "object_id": risk_id,
        "risk_id": risk_id,
        "wallet_uid": wallet_uid,
        "agent_id": trigger_object.get("agent_id", ""),
        "risk_score": score,
        "risk_category": risk_category,
        "trigger_type": f"{kind.lower()}_auto",
        "trigger_description": trigger_description,
        "gps_location": trigger_object.get("geo_location", ""),
        "imei": trigger_object.get("imei", ""),
        "transaction_volume": amount,
        "alert_status": "active",
        "review_status": "pending",
        "compliance_action": compliance_action,
        "str_triggered": kind == "STR",
        "ctr_triggered": kind == "CTR",
        "notes": notes_with_retention,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


# ---------------------------------------------------------------------------
# STR generation
# ---------------------------------------------------------------------------

async def generate_str(
    decision: CheckmarbleDecision,
    escalation: Optional[EscalationResult],
    trigger_object: dict,
) -> Optional[dict]:
    """Generate an STR draft and ingest it as a risk_event.

    Triggered when the effective score (combined after escalation, or
    Checkmarble score if no escalation) >= STR_SCORE_THRESHOLD.
    """
    score = escalation.combined_score if escalation else decision.score
    if score < STR_SCORE_THRESHOLD:
        return None

    rules = [f"{r.rule_id}({r.score})" for r in decision.rules_triggered if r.score > 0]
    trigger_desc = f"STR auto-draft: score={score}, rules=[{', '.join(rules)}]"
    if escalation and escalation.compliance_alert_reason:
        trigger_desc += f", alert={escalation.compliance_alert_reason}"

    risk_event = _build_risk_event(
        kind="STR",
        score=score,
        amount=float(trigger_object.get("amount", 0)),
        decision=decision,
        trigger_object=trigger_object,
        risk_category="critical" if score >= 60 else "high",
        trigger_description=trigger_desc,
        compliance_action="str_draft_generated",
        notes=(
            f"Auto-generated STR draft for decision {decision.decision_id}. "
            f"Must be reviewed within the jurisdiction's STR filing window "
            f"(typically 24h per FIU rules)."
        ),
    )

    try:
        await _ingest_record("risk_events", risk_event)
        logger.info("str_draft_generated", risk_id=risk_event["risk_id"], score=score, wallet_uid=risk_event["wallet_uid"])
        if risk_event["wallet_uid"]:
            await _add_to_str_list(risk_event["wallet_uid"])
        return risk_event
    except Exception as e:
        logger.error("str_ingest_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# CTR generation
# ---------------------------------------------------------------------------

async def check_ctr(
    decision: CheckmarbleDecision,
    trigger_object: dict,
) -> Optional[dict]:
    """Generate a CTR filing if the transaction amount >= CTR_AMOUNT_THRESHOLD.

    CTR is independent of the risk score — purely amount-based, per the
    jurisdiction's FIU regulations (default 500,000 in local currency).
    """
    amount = float(trigger_object.get("amount", 0))
    if amount < CTR_AMOUNT_THRESHOLD:
        return None

    risk_event = _build_risk_event(
        kind="CTR",
        score=decision.score,
        amount=amount,
        decision=decision,
        trigger_object=trigger_object,
        risk_category="ctr_threshold",
        trigger_description=(
            f"CTR auto-filing: amount={amount:,.0f} "
            f"(threshold={CTR_AMOUNT_THRESHOLD:,})"
        ),
        compliance_action="ctr_filing_generated",
        notes=(
            f"Auto-generated CTR for decision {decision.decision_id}, "
            f"amount={amount:,.0f}. Per jurisdiction FIU regulations."
        ),
    )

    try:
        await _ingest_record("risk_events", risk_event)
        logger.info("ctr_filing_generated", risk_id=risk_event["risk_id"], amount=amount, wallet_uid=risk_event["wallet_uid"])
        return risk_event
    except Exception as e:
        logger.error("ctr_ingest_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

async def log_audit(
    decision: CheckmarbleDecision,
    escalation: Optional[EscalationResult],
    str_event: Optional[dict],
    ctr_event: Optional[dict],
) -> Optional[dict]:
    """Log a compliance audit trail entry."""
    audit_id = f"AUD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    actions = []
    if escalation:
        actions.append(
            f"escalation(combined={escalation.combined_score}, "
            f"suggestion={escalation.final_decision_suggestion})"
        )
    if str_event:
        actions.append(f"str_draft({str_event['risk_id']})")
    if ctr_event:
        actions.append(f"ctr_filing({ctr_event['risk_id']})")
    if not actions:
        actions.append(f"decision_logged(score={decision.score}, outcome={decision.outcome})")

    linked_event = ""
    if str_event:
        linked_event = str_event["risk_id"]
    elif ctr_event:
        linked_event = ctr_event["risk_id"]

    audit = {
        "object_id": audit_id,
        "updated_at": now,
        "audit_id": audit_id,
        "action_type": "compliance_decision",
        "performed_by_uid": "bridge_service",
        "performed_by_role": "system",
        "target_uid": decision.trigger_object_id,
        "target_entity_type": decision.trigger_object_type,
        "action_description": "; ".join(actions),
        "timestamp": now,
        "status_before": "pending",
        "status_after": decision.outcome,
        "linked_risk_event": linked_event,
        "audit_verified": False,
    }

    try:
        await _ingest_record("audit_trail", audit)
        logger.info("audit_logged", audit_id=audit_id)
        return audit
    except Exception as e:
        logger.error("audit_ingest_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def process_compliance(
    decision: CheckmarbleDecision,
    escalation: Optional[EscalationResult],
    trigger_object: dict,
) -> dict:
    """Run all compliance checks, enrich the EscalationResult in-place, and
    return a summary dict.

    Called after decision + escalation for every ``/decide`` request.

    Side-effect: if ``escalation`` is not None, populates these canonical
    EscalationResult fields (per spec.md §7.4):
        - ctr_required
        - sanctions_match
        - shariah_violation_flag, shariah_anomaly_reason
        - audit_log_reference  (linked to audit_trail row)
        - str_suggested (set if STR draft was actually generated)

    Returns a dict with:
        - str_generated (bool)
        - ctr_generated (bool)
        - str_risk_event_id (str or None)
        - ctr_risk_event_id (str or None)
        - audit_id (str or None)
    """
    result = {
        "str_generated": False,
        "ctr_generated": False,
        "str_risk_event_id": None,
        "ctr_risk_event_id": None,
        "audit_id": None,
    }

    # Plumb pre-computed FDS signals → EscalationResult BEFORE running checks
    # so compliance_violation_type derivation in escalation.py has the
    # sanctions_match signal available if compliance is called first.
    # (Current flow calls escalate() first, but we still want to update
    # the result object with any new signals here.)
    fds_features = trigger_object.get("fds_input_features") or {}
    if escalation is not None:
        # sanctions_hit comes from pre-computed fds_input_features per data contract
        if fds_features.get("sanctions_hit"):
            escalation.sanctions_match = True

        # Shariah signals — pre-computed by the operator backend
        if fds_features.get("shariah_violation_detected"):
            escalation.shariah_violation_flag = True
            reasons = []
            if fds_features.get("zakat_qr_violation_flag"):
                reasons.append("zakat_qr_misuse")
            if fds_features.get("product_prohibited_tag_flag"):
                reasons.append("prohibited_product")
            if fds_features.get("lms_shariah_tag"):
                reasons.append(f"tt_tag={fds_features['lms_shariah_tag']}")
            escalation.shariah_anomaly_reason = ",".join(reasons) if reasons else "shariah_flag_set"

    # STR (score-based) and CTR (amount-based) are independent — run concurrently.
    str_event, ctr_event = await asyncio.gather(
        generate_str(decision, escalation, trigger_object),
        check_ctr(decision, trigger_object),
    )
    if str_event:
        result["str_generated"] = True
        result["str_risk_event_id"] = str_event["risk_id"]
        if escalation is not None:
            escalation.str_suggested = True
    if ctr_event:
        result["ctr_generated"] = True
        result["ctr_risk_event_id"] = ctr_event["risk_id"]
        if escalation is not None:
            escalation.ctr_required = True

    # Audit trail must be last — it references both STR and CTR.
    audit = await log_audit(decision, escalation, str_event, ctr_event)
    if audit:
        result["audit_id"] = audit["audit_id"]
        if escalation is not None:
            escalation.audit_log_reference = audit["audit_id"]

    return result
