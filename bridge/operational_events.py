"""the operator operational events + approved customer-facing status strings.

This module implements the the platform's compliance hard-lock policy risk-band remap (§4.1, §M1) and
the approved-strings whitelist (§4.2). It is the single source of truth for:

  • the four operational events that wrap every Marble FDS outcome, and
  • the five and only five strings that may appear on customer-facing,
    agent-facing, or merchant-facing surfaces in response to an FDS event.

The mapping deliberately replaces Marble's default ``decline`` (which would
otherwise imply auto-block) with ``OPERATIONAL_HOLD_CHECKER`` — an event
that REQUIRES human Compliance Checker action (the platform's compliance hard-lock policy §M1
hard-lock). Marble FDS at the operator never auto-blocks; the only effect of
a Critical-band score is to *queue* the case for two-person human approval.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


class OperationalEvent(str, Enum):
    """The four operational events that wrap every Marble FDS outcome.

    Per the platform's compliance hard-lock policy §4.1 (M1):
      • PASS_THROUGH                 (score 0–24)
      • SILENT_REVIEW                (score 25–39)
      • OPERATIONAL_HOLD_MAKER       (score 40–59)
      • OPERATIONAL_HOLD_CHECKER     (score 60+)

    The transition from Marble's default ``decline`` (auto-block) at 60+ to
    ``OPERATIONAL_HOLD_CHECKER`` (human review required) is the platform's
    M1 hard-lock — the bridge produces the event but performs no automated
    account action.
    """
    PASS_THROUGH = "PASS_THROUGH"
    SILENT_REVIEW = "SILENT_REVIEW"
    OPERATIONAL_HOLD_MAKER = "OPERATIONAL_HOLD_MAKER"
    OPERATIONAL_HOLD_CHECKER = "OPERATIONAL_HOLD_CHECKER"


def map_score_to_event(score: float | None) -> OperationalEvent:
    """Map a 0–100 combined risk score to the corresponding event.

    Boundaries per the platform's compliance hard-lock policy §4.1:
        score < 25  → PASS_THROUGH
        score < 40  → SILENT_REVIEW
        score < 60  → OPERATIONAL_HOLD_MAKER
        score ≥ 60  → OPERATIONAL_HOLD_CHECKER
    """
    s = float(score or 0)
    if s < 25:
        return OperationalEvent.PASS_THROUGH
    if s < 40:
        return OperationalEvent.SILENT_REVIEW
    if s < 60:
        return OperationalEvent.OPERATIONAL_HOLD_MAKER
    return OperationalEvent.OPERATIONAL_HOLD_CHECKER


# ─── Approved customer-facing status strings (§4.2 hard-lock) ──────────
#
# These are the ONLY phrases permitted on any customer/agent/merchant
# surface in response to a Marble FDS event. The use of any other AML or
# compliance terminology constitutes a tipping-off breach under the operator
# v20.6 §2.4.5 and the Afghan AML/CFT Regulation.

class ApprovedCustomerString(str, Enum):
    """The five and only five approved neutral strings (§4.2)."""
    UNDER_REVIEW = "Under review"
    ACTION_REQUIRED = "Action required — please contact support"
    CURRENTLY_UNAVAILABLE = "Currently unavailable"
    PROCESSING = "Processing — please wait"
    REQUEST_NOT_COMPLETED = "Request could not be completed"


APPROVED_STRINGS: Final[frozenset[str]] = frozenset(
    s.value for s in ApprovedCustomerString
)


# ─── Forbidden tipping-off vocabulary ──────────────────────────────────
# Per the platform's compliance hard-lock policy §4.2 final paragraph: any synonym or implication of
# these terms on a customer/agent/merchant surface is a tipping-off breach.

FORBIDDEN_CUSTOMER_TERMS: Final[frozenset[str]] = frozenset({
    "aml", "anti-money laundering", "money laundering",
    "suspicious", "investigation", "investigated",
    "str", "ctr", "lctr",
    "compliance risk", "compliance hold",
    "hold reason", "risk score", "risk band",
    "fraud", "fraudulent", "flagged", "watchlist",
    "blocked for compliance reasons", "regulatory hold",
})


def event_to_customer_string(event: OperationalEvent) -> ApprovedCustomerString:
    """Map an operational event to the single approved customer-facing string.

    Per the platform's compliance hard-lock policy §4.2:
      PASS_THROUGH              → no message (transaction proceeds normally)
      SILENT_REVIEW             → no message (background review only)
      OPERATIONAL_HOLD_MAKER    → "Under review"
      OPERATIONAL_HOLD_CHECKER  → "Under review"

    SILENT_REVIEW callers can render PROCESSING during the review window
    if they need a status, but no AML detail; PASS_THROUGH should render
    the channel's normal success state. By default, both return PROCESSING
    so the caller doesn't have to special-case the absent-string cases.
    """
    if event in (OperationalEvent.OPERATIONAL_HOLD_MAKER,
                 OperationalEvent.OPERATIONAL_HOLD_CHECKER):
        return ApprovedCustomerString.UNDER_REVIEW
    return ApprovedCustomerString.PROCESSING


# ─── Role-gated response visibility (§M4 / §11.2) ─────────────────────
# Roles permitted to see full FDS scoring detail in API responses.
# Maps to the platform's compliance hard-lock policy §11.1 visibility matrix: Compliance Maker,
# Compliance Checker, CCO, and Internal Audit have full visibility;
# everyone else (Operations, Finance, DevOps, agents, merchants,
# customers) sees only operational_event + customer_status_string.

COMPLIANCE_ROLES: Final[frozenset[str]] = frozenset({
    "COMPLIANCE_MAKER",
    "COMPLIANCE_CHECKER",
    "CCO",
    "CHIEF_COMPLIANCE_OFFICER",
    "INTERNAL_AUDIT",
})


def is_compliance_caller(role_header_value: str | None) -> bool:
    """True if the caller's X-Compliance-Role header value is on the
    whitelist permitted to see full FDS scoring detail."""
    if not role_header_value:
        return False
    return role_header_value.strip().upper() in COMPLIANCE_ROLES


def build_minimal_response(
    transaction_id: str,
    operational_event: OperationalEvent,
    customer_status_string: ApprovedCustomerString,
) -> dict:
    """The customer-facing response shape (§M4). NEVER includes scoring,
    rules, SHAP values, or any AML reasoning. Only the neutral status
    string and the operational event tag (which itself is intentionally
    abstract — no AML or risk vocabulary)."""
    return {
        "transaction_id": transaction_id,
        "operational_event": operational_event.value,
        "status": customer_status_string.value,
    }


def assert_customer_string_safe(text: str) -> None:
    """Hard-lock check — raise ValueError if a string contains a forbidden
    tipping-off term. Intended for use anywhere a string is about to leave
    the bridge bound for a customer/agent/merchant surface.
    """
    if not text:
        return
    lowered = text.lower()
    for term in FORBIDDEN_CUSTOMER_TERMS:
        if term in lowered:
            raise ValueError(
                f"String '{text!r}' contains forbidden tipping-off term "
                f"'{term}' — refuse per the platform's compliance hard-lock policy §4.2."
            )
