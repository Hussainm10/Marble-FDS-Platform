"""Marbel + GNN escalation logic with combined scoring.

Produces a canonical ``EscalationResult`` per spec.md §7.4:
    risk_score, risk_level, triggered_rules, latency_ms, timestamp,
    escalation_path_id, compliance_violation_type, etc.

Populated here (escalation.py): score layers, risk_level, decision
suggestion, triggered_rules list, latency_ms, timestamp, escalation_path_id,
compliance_violation_type.

Populated downstream (compliance.py): ctr_required, sanctions_match,
shariah_violation_flag, shariah_anomaly_reason, audit_log_reference.
"""

import math
import uuid
from datetime import datetime
from typing import Optional

import httpx
import structlog
from asgi_correlation_id import correlation_id

from .config import settings
from .models import (
    CheckmarbleDecision,
    ComplianceViolationType,
    EscalationResult,
    FinalDecisionSuggestion,
    RiskLevel,
)

# Feature contract that the GNN v4 service expects (PaySim training).
# Keeping it explicit here so a schema drift in the model registers as a 400
# at the GNN service rather than as silently-wrong inference.
_GNN_V4_FEATURE_DIM = 42

# Mobile-money transaction_type → PaySim taxonomy. Anything outside this map
# falls through as all-zero one-hot (the model treats it as out-of-vocab).
_PAYSIM_TYPE_MAP = {
    "transfer": "TRANSFER",
    "cash_out": "CASH_OUT",
    "cashout": "CASH_OUT",
    "cash_in": "CASH_IN",
    "cashin": "CASH_IN",
    "payment": "PAYMENT",
    "debit": "DEBIT",
}
_PAYSIM_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

logger = structlog.get_logger("bridge.escalation")


def _correlation_headers() -> dict:
    cid = correlation_id.get()
    return {"X-Correlation-ID": cid} if cid else {}


def _risk_level(score: float) -> RiskLevel:
    """Map a 0-100 score to the canonical 4-tier RiskLevel (spec.md §1.3)."""
    if score <= settings.SCORE_APPROVE_MAX:      # 0-24
        return RiskLevel.LOW
    if score <= settings.SCORE_REVIEW_MAX:       # 25-39
        return RiskLevel.MEDIUM
    if score <= settings.SCORE_BLOCK_MAX:        # 40-59
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL                    # 60+


def _compliance_violation_type(
    triggered_rule_ids: list[str],
    sanctions_match: bool = False,
) -> ComplianceViolationType:
    """Derive the dominant compliance-violation category from triggered rules.

    Priority (highest first) — maps to typical compliance-officer queue routing:
        Sanctions > Shariah > CTR > STR > Velocity > Device > AML > None

    Args:
        triggered_rule_ids: rule IDs that fired (e.g. 'RSK_VEL_060_velocity_spike')
        sanctions_match: explicit signal from fds_input_features.sanctions_hit
                         (overrides rule-ID inference if True)
    """
    if sanctions_match:
        return ComplianceViolationType.SANCTIONS

    # Rule-ID inference — check specific prefixes/keywords in priority order.
    # Each tier returns early so the highest-severity match wins.
    joined = " ".join(triggered_rule_ids).upper()

    if any(rid.startswith("SHR_") for rid in triggered_rule_ids) or "SHARIAH" in joined or "ZAKAT" in joined:
        return ComplianceViolationType.SHARIAH

    if "SANCTIONS" in joined:
        return ComplianceViolationType.SANCTIONS

    if any(rid.startswith("CMP_CTR") for rid in triggered_rule_ids) or "CTR_" in joined:
        return ComplianceViolationType.CTR

    if any(rid.startswith("CMP_STR") for rid in triggered_rule_ids) or "STR_" in joined:
        return ComplianceViolationType.STR

    if "VEL" in joined or "VELOCITY" in joined:
        return ComplianceViolationType.VELOCITY

    if "IMEI" in joined or "DEV" in joined or "DEVICE" in joined:
        return ComplianceViolationType.DEVICE

    if "AML" in joined or "BLACKLIST" in joined:
        return ComplianceViolationType.AML

    return ComplianceViolationType.NONE


async def _call_ml_layer(layer_name: str, url: str, payload: dict) -> Optional[dict]:
    """POST to an ML evaluator (Marbel or GNN). Returns the JSON body or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=_correlation_headers()) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.warning("ml_call_failed", layer=layer_name, error=str(e))
        return None
    except Exception:
        logger.error("ml_call_unexpected", layer=layer_name, exc_info=True)
        return None


async def _call_marbel(decision: CheckmarbleDecision) -> Optional[dict]:
    payload = {
        "decision_id": decision.decision_id,
        "checkmarble_score": decision.score,
        "trigger_object_type": decision.trigger_object_type,
        "trigger_object_id": decision.trigger_object_id,
        "rules_triggered": [
            {"rule_id": r.rule_id, "score": r.score, "description": r.description}
            for r in decision.rules_triggered
        ],
    }
    return await _call_ml_layer("Marbel", f"{settings.MARBEL_API_URL}/evaluate", payload)


def _build_node_features_v4(trigger_object: Optional[dict]) -> Optional[list[float]]:
    """Build the GNN v4 input vector (42 floats) from a single transaction.

    PaySim training features are aggregations over an account's *full* history
    (count of txns by sender, sum/mean/std of amounts, etc.). At decision time
    we only see the current transaction, so most aggregations degenerate to
    single-value stats: count=1, std=0, sum=mean=min=max=amount, etc. The model
    still produces a usable signal because the type one-hots, balance fields,
    and graph-degree priors carry information.

    Returns None if the trigger object is missing or malformed — caller falls
    back to GNN's stub mode.

    For higher fidelity in production, replace the single-txn aggregations
    here with a feature-store / Checkmarble history lookup keyed by
    from_wallet_uid / to_wallet_uid.
    """
    if not trigger_object:
        return None
    try:
        amt = float(trigger_object.get("amount") or 0)
        old_bal_orig = float(trigger_object.get("oldbalanceOrg") or 0)
        new_bal_orig = float(trigger_object.get("newbalanceOrig") or 0)
        old_bal_dest = float(trigger_object.get("oldbalanceDest") or 0)
        new_bal_dest = float(trigger_object.get("newbalanceDest") or 0)

        # PaySim's "step" is hours since dataset start (0..743). At inference we
        # don't have a meaningful absolute step, so we use hour-of-day from the
        # timestamp as a stable proxy (0..23 ⊆ 0..743).
        step = 0.0
        ts = trigger_object.get("timestamp")
        if ts:
            try:
                step = float(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).hour)
            except (ValueError, AttributeError):
                step = 0.0

        # Transaction-type one-hot. Same single transaction shows up as both
        # send and recv for the model (sender + receiver perspective).
        raw_type = (trigger_object.get("transaction_type") or "").lower()
        paysim_type = _PAYSIM_TYPE_MAP.get(raw_type)
        type_onehot = [1.0 if t == paysim_type else 0.0 for t in _PAYSIM_TYPES]

        # Single-txn aggregations: count=1, sum=amount, mean=amount, std=0, min=max=amount.
        send_amt = [1.0, amt, amt, 0.0, amt, amt]
        recv_amt = [1.0, amt, amt, 0.0, amt, amt]
        send_oldbal = [old_bal_orig, old_bal_orig]
        send_newbal = [new_bal_orig, new_bal_orig]
        recv_oldbal = [old_bal_dest, old_bal_dest]
        recv_newbal = [new_bal_dest, new_bal_dest]
        send_step = [step, step, 1.0]
        recv_step = [step, step, 1.0]

        # Graph degree priors for a single observed edge: in=out=1, total=2.
        # in_out_ratio matches the training script's: in / (out + 1).
        in_deg, out_deg = 1.0, 1.0
        graph = [
            in_deg,
            out_deg,
            in_deg + out_deg,
            in_deg / (out_deg + 1),
            math.log1p(in_deg),
            math.log1p(out_deg),
        ]

        features = (
            send_amt + send_oldbal + send_newbal + send_step
            + recv_amt + recv_oldbal + recv_newbal + recv_step
            + type_onehot + type_onehot
            + graph
        )
        if len(features) != _GNN_V4_FEATURE_DIM:
            logger.warning("gnn_feature_build_dim_mismatch",
                           built=len(features), expected=_GNN_V4_FEATURE_DIM)
            return None
        return features
    except (TypeError, ValueError) as e:
        logger.warning("gnn_feature_build_failed", error=str(e))
        return None


async def _call_gnn(
    decision: CheckmarbleDecision,
    marbel_result: dict,
    trigger_object: Optional[dict] = None,
) -> Optional[dict]:
    payload = {
        "decision_id": decision.decision_id,
        "checkmarble_score": decision.score,
        "marbel_score": marbel_result.get("marbel_risk_score", 0),
        "trigger_object_type": decision.trigger_object_type,
        "trigger_object_id": decision.trigger_object_id,
        "entity_id": decision.trigger_object_id,
    }
    features = _build_node_features_v4(trigger_object)
    if features is not None:
        payload["node_data"] = {"features": features, "neighbors": []}
    return await _call_ml_layer("GNN", f"{settings.GNN_API_URL}/evaluate", payload)


def _merge_scores(
    checkmarble_score: float,
    marbel_score: Optional[float],
    gnn_score: Optional[float],
) -> float:
    """Merge scores from all three layers.

    Weighting: Checkmarble 40%, Marbel 35%, GNN 25%
    If a layer is unavailable, redistribute proportionally.
    """
    weights = {"checkmarble": 0.40, "marbel": 0.35, "gnn": 0.25}
    total_weight = weights["checkmarble"]
    weighted_sum = checkmarble_score * weights["checkmarble"]

    if marbel_score is not None:
        total_weight += weights["marbel"]
        weighted_sum += marbel_score * weights["marbel"]

    if gnn_score is not None:
        total_weight += weights["gnn"]
        weighted_sum += gnn_score * weights["gnn"]

    if total_weight == 0:
        return checkmarble_score

    return round(weighted_sum / total_weight, 1)


async def escalate(
    decision: CheckmarbleDecision,
    trigger_object: Optional[dict] = None,
) -> EscalationResult:
    """Escalate a Checkmarble decision through Marbel and optionally GNN.

    Called when Checkmarble score >= MARBEL_ESCALATION_THRESHOLD (default 40).
    Returns an EscalationResult populated per spec.md §7.4.

    ``trigger_object`` is the raw transaction dict from the /decide caller. If
    provided, it's used to build node features for the GNN's real inference
    path (otherwise GNN stays in stub_no_features mode).

    latency_ms is NOT set here — the caller in bridge/app.py /decide wraps the
    full pipeline (decide + escalate + compliance) with a wall-clock timer and
    assigns latency_ms on the returned result.
    """
    escalation_path_id = str(uuid.uuid4())

    logger.info("escalate_start",
                decision_id=decision.decision_id,
                score=decision.score,
                outcome=decision.outcome,
                path_id=escalation_path_id)

    result = EscalationResult(
        checkmarble_score=decision.score,
        combined_score=decision.score,
        risk_score=decision.score,
        risk_level=_risk_level(decision.score),
        escalation_path_id=escalation_path_id,
    )

    marbel_result = await _call_marbel(decision)
    if marbel_result:
        result.marbel_triggered = True
        marbel_score = marbel_result.get("marbel_risk_score", 0)
        result.marbel_score = marbel_score
        result.feature_contributions = marbel_result.get("feature_contributions", {})

        if marbel_score >= settings.GNN_ESCALATION_THRESHOLD:
            gnn_result = await _call_gnn(decision, marbel_result, trigger_object)
            if gnn_result:
                result.gnn_triggered = True
                result.gnn_score = gnn_result.get("enhanced_score", 0)
                result.entity_graph_signals = gnn_result.get("entity_graph_signals", {})

    merged = _merge_scores(
        result.checkmarble_score, result.marbel_score, result.gnn_score
    )
    result.combined_score = merged
    result.risk_score = merged
    result.risk_level = _risk_level(merged)

    # Map risk band → suggestion + AML/STR flags. Per GNN PDF §2.1 and spec §7.4.
    # SoftHold (partial-confidence cases) is reserved for future use.
    if merged >= 60:
        result.final_decision_suggestion = FinalDecisionSuggestion.ESCALATE
        result.aml_risk_flag = True
        result.str_suggested = True
    elif merged >= 40:
        result.final_decision_suggestion = FinalDecisionSuggestion.ESCALATE
        result.aml_risk_flag = merged >= 50
    elif merged >= 25:
        result.final_decision_suggestion = FinalDecisionSuggestion.MONITOR
    else:
        result.final_decision_suggestion = FinalDecisionSuggestion.ALLOW

    # ── the operator operational event remap (the platform's compliance hard-lock policy §4.1, §M1) ──
    # M1 hard-lock: replace Marble's auto-block default with an event that
    # requires human Compliance Officer action. The bridge produces the
    # event but never enforces it — enforcement is the operator's back-office.
    from .operational_events import (
        map_score_to_event, event_to_customer_string,
    )
    op_event = map_score_to_event(merged)
    result.operational_event = op_event.value
    result.customer_status_string = event_to_customer_string(op_event).value

    rule_names = [r.rule_id for r in decision.rules_triggered if r.score > 0]
    result.triggered_rules = list(rule_names)

    # compliance_alert_reason stays human-readable (may include graph signals);
    # triggered_rules is strictly rule IDs per the canonical contract.
    reason_parts = list(rule_names)
    if result.gnn_triggered:
        signals = result.entity_graph_signals
        shared = signals.get("shared_devices", 0)
        linked = signals.get("linked_wallets", 0)
        if shared or linked:
            reason_parts.append(f"SharedDevices:{shared}")
            reason_parts.append(f"LinkedWallets:{linked}")
    result.compliance_alert_reason = "+".join(reason_parts) if reason_parts else ""

    result.compliance_violation_type = _compliance_violation_type(
        result.triggered_rules,
        sanctions_match=result.sanctions_match,
    )

    logger.info("escalate_complete",
                path_id=result.escalation_path_id,
                combined_score=result.combined_score,
                risk_level=result.risk_level.value,
                suggestion=result.final_decision_suggestion.value,
                violation_type=result.compliance_violation_type.value)

    return result
