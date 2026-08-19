"""Ingestion-time payload validators.

Hard-locks per the platform's compliance hard-lock policy §8 (M5 — MSISDN exclusion) and §9/12 (M12 —
no card data through the bridge until a written PCI data-flow agreement is
in place). Both validators run before any payload reaches Marble.

These validators reject payloads at the bridge boundary; nothing forbidden
is ever forwarded to Marble's data model. Violations raise
``ForbiddenFieldError`` (HTTP 400 via the canonical FDS error handler).
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .errors import FDSError


# ---------------------------------------------------------------------------
# M12 — card-data fields are categorically forbidden
# ---------------------------------------------------------------------------
# Per the platform's compliance hard-lock policy §M12: "No card-flow (PAN, CVV2, PIN block, track data)
# permitted through Marble Bridge until written data-flow agreement with
# acquirer/issuer executed. Tokens only."
FORBIDDEN_CARD_FIELDS: frozenset[str] = frozenset({
    # Primary Account Number variants
    "pan", "primary_account_number",
    "card_no", "cardno", "card_number", "cardnumber",
    "credit_card_number", "debit_card_number",
    # CVV / CVC / CID
    "cvv", "cvv2", "cvc", "cvc2", "cid", "card_verification_value",
    "card_security_code",
    # PIN block / encrypted PIN
    "pin", "pin_block", "encrypted_pin", "pin_encrypted",
    # Magstripe track data (ISO 7813)
    "track1", "track_1", "track2", "track_2", "track3", "track_3",
    "track_data", "magstripe", "magstripe_data",
    # EMV chip data
    "emv_data", "icc_data",
})


# ---------------------------------------------------------------------------
# M5 — MSISDN cannot be a primary/aggregation key
# ---------------------------------------------------------------------------
# Per the platform's compliance hard-lock policy §8.1 / §8.2: MSISDN binds to UID at authentication time
# and may be rebound (SIM-recycle resilience). Using MSISDN as an AML
# aggregation key would attribute Customer B's activity to Customer A.
#
# Allowed: MSISDN as a *delivery channel reference* in biometric_logs and
# notification_logs (only).
# Forbidden everywhere else: MSISDN-shaped values as ``object_id`` (the
# primary key), or as the value of any field whose name implies "key" or
# "aggregation".

_MSISDN_KEY_FIELD_NAMES: frozenset[str] = frozenset({
    "msisdn", "phone", "phone_number", "mobile", "mobile_number",
    "primary_key", "aggregation_key", "aml_key",
})

# E.164-ish: optional +, 8–15 digits. Conservative bound covering most
# national MSISDN formats including country code.
_MSISDN_RE = re.compile(r"^\+?\d{8,15}$")

# Tables in which MSISDN may legitimately appear as a delivery channel field
# (NOT as the primary key — only as a non-key attribute). Per the platform's compliance hard-lock policy
# §M5 final paragraph.
TABLES_ALLOWING_MSISDN_AS_CHANNEL: frozenset[str] = frozenset({
    "biometric_logs", "notification_logs",
})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ForbiddenFieldError(FDSError):
    """Raised when an ingest payload contains a field forbidden by
    the platform's compliance hard-lock policy §M5 (MSISDN) or §M12 (card data).

    Canonical code DOC-415 (invalid document) at HTTP 400 (Bad Request) —
    the payload is rejected at the bridge boundary before any backend
    interaction. Caller should fix the payload and retry.
    """
    code = "DOC-415"
    http_status = 400


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _walk_keys(obj: Any) -> Iterable[tuple[str, Any]]:
    """Yield (key_lower, value) for every dict key in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k.lower(), v
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


def _looks_like_msisdn(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return _MSISDN_RE.match(str(int(value))) is not None
    if isinstance(value, str):
        return _MSISDN_RE.match(value.strip()) is not None
    return False


def reject_card_data(payload: dict) -> None:
    """Raise ``ForbiddenFieldError`` if any card-data field is present.

    Hard-lock per the platform's compliance hard-lock policy §M12. No card-flow data may transit the
    bridge until a written data-flow agreement is executed.
    """
    if not isinstance(payload, dict):
        return
    for k, _ in _walk_keys(payload):
        if k in FORBIDDEN_CARD_FIELDS:
            raise ForbiddenFieldError(
                message="Card-flow data is forbidden through this bridge "
                        "(the platform's compliance hard-lock policy §M12). Tokens only.",
                details={"forbidden_field": k},
            )


def reject_msisdn_as_key(table_name: str, object_id: str | None,
                         payload: dict) -> None:
    """Raise ``ForbiddenFieldError`` if MSISDN appears as a primary or
    aggregation key.

    Per the platform's compliance hard-lock policy §M5:
    - ``object_id`` may never be an MSISDN-shaped value (it is the primary key).
    - Any field whose name implies "key" or "aggregation" must not contain
      an MSISDN-shaped value, regardless of table.
    - ``biometric_logs`` and ``notification_logs`` are allowed to carry
      MSISDN as a delivery-channel attribute (a non-key field).
    """
    # Rule 1: object_id never MSISDN-shaped
    if object_id and _looks_like_msisdn(object_id):
        raise ForbiddenFieldError(
            message="MSISDN-shaped values may not be used as object_id "
                    "(the platform's compliance hard-lock policy §M5). Use entity_id or uid_id.",
            details={"table": table_name, "object_id": object_id},
        )

    if not isinstance(payload, dict):
        return

    table_lower = (table_name or "").lower()
    in_channel_table = table_lower in TABLES_ALLOWING_MSISDN_AS_CHANNEL

    # Rule 2: any "key"-named field carrying an MSISDN — always forbidden.
    # Rule 3: in non-channel tables, an MSISDN field at all is treated as
    # an aggregation-key risk (per spirit of §8.1 last paragraph).
    for k, v in _walk_keys(payload):
        if k in _MSISDN_KEY_FIELD_NAMES:
            if _looks_like_msisdn(v) and not (in_channel_table and k in {"msisdn", "phone", "phone_number", "mobile", "mobile_number"}):
                raise ForbiddenFieldError(
                    message=f"Field '{k}' carries an MSISDN-shaped value, "
                            f"which is forbidden as an AML key per "
                            f"the platform's compliance hard-lock policy §M5.",
                    details={"table": table_name, "field": k},
                )


def validate_ingest_payload(table_name: str, object_id: str | None,
                            payload: dict) -> None:
    """Apply all hard-lock validators to an inbound ingest payload."""
    reject_card_data(payload)
    reject_msisdn_as_key(table_name, object_id, payload)
