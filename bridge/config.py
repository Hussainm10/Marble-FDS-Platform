"""Configuration from environment variables."""

import os


class Settings:
    # Checkmarble
    CHECKMARBLE_API_URL: str = os.getenv("CHECKMARBLE_API_URL", "http://localhost:8080")
    CHECKMARBLE_API_KEY: str = os.getenv("CHECKMARBLE_API_KEY", "")

    # Marbel ML Engine
    MARBEL_API_URL: str = os.getenv("MARBEL_API_URL", "http://marbel:5000")
    MARBEL_ESCALATION_THRESHOLD: int = int(os.getenv("MARBEL_ESCALATION_THRESHOLD", "40"))

    # GNN Graph Fraud Engine
    GNN_API_URL: str = os.getenv("GNN_API_URL", "http://gnn:5001")
    GNN_ESCALATION_THRESHOLD: int = int(os.getenv("GNN_ESCALATION_THRESHOLD", "60"))

    # Bridge service
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    HOST: str = os.getenv("BRIDGE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BRIDGE_PORT", "8000"))

    # Optional production hardening (unset in dev = off)
    BRIDGE_API_KEY: str = os.getenv("BRIDGE_API_KEY", "")      # client auth on bridge endpoints
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")      # HMAC verification on /webhooks/checkmarble
    DEBUG_DETAILS: bool = os.getenv("DEBUG_DETAILS", "false").lower() in ("1", "true", "yes")

    # ── Response visibility (the platform's compliance hard-lock policy §M4 / §11.2) ───────────────
    # FDS scoring detail (raw score, rules triggered, layer scores, SHAP,
    # GNN signals) MUST NOT appear in the response body returned to
    # customer/agent/merchant channels. Compliance roles receive the full
    # payload via the X-Compliance-Role header (whitelist enforced).
    #
    # Modes:
    #   "minimal" — production. Default response is customer-facing (only
    #               operational_event + customer_status_string + txn_id);
    #               full payload only when X-Compliance-Role is whitelisted.
    #   "verbose" — dev/test. Full payload always returned (header ignored).
    BRIDGE_RESPONSE_MODE: str = os.getenv("BRIDGE_RESPONSE_MODE", "verbose").lower()

    # Risk band thresholds
    SCORE_APPROVE_MAX: int = 24
    SCORE_REVIEW_MAX: int = 39
    SCORE_BLOCK_MAX: int = 59
    # 60+ = decline

    # ── Audit retention (platform compliance hard-lock policy) ────────────
    # Per common central-bank AML/CFT retention floors:
    #   • General audit_trail records:      ≥ 7 years  (2557 days)
    #   • AML-linked records (risk_events,
    #     STR/CTR drafts, sanction matches): ≥ 10 years (3650 days)
    # These are minimums. Operators may set higher via env, never lower —
    # the post-init validation rejects sub-minimum values.
    AUDIT_RETENTION_DAYS_GENERAL: int = int(
        os.getenv("AUDIT_RETENTION_DAYS_GENERAL", "2557")
    )
    AUDIT_RETENTION_DAYS_AML: int = int(
        os.getenv("AUDIT_RETENTION_DAYS_AML", "3650")
    )

    # Regulatory minimums — hard-coded floors. Used by validation below.
    _AUDIT_RETENTION_MIN_GENERAL_DAYS: int = 2557   # 7 years (the platform's compliance hard-lock policy)
    _AUDIT_RETENTION_MIN_AML_DAYS: int = 3650        # 10 years (the platform's compliance hard-lock policy)

    def __init__(self):
        # Enforce regulatory retention minimums at boot. Misconfiguration
        # here means audit data could be discarded before the regulator
        # requires — refuse to start rather than silently violate policy.
        if self.AUDIT_RETENTION_DAYS_GENERAL < self._AUDIT_RETENTION_MIN_GENERAL_DAYS:
            raise ValueError(
                f"AUDIT_RETENTION_DAYS_GENERAL must be >= "
                f"{self._AUDIT_RETENTION_MIN_GENERAL_DAYS} days (7 years) per "
                f"the platform's compliance hard-lock policy §M6 — got {self.AUDIT_RETENTION_DAYS_GENERAL}."
            )
        if self.AUDIT_RETENTION_DAYS_AML < self._AUDIT_RETENTION_MIN_AML_DAYS:
            raise ValueError(
                f"AUDIT_RETENTION_DAYS_AML must be >= "
                f"{self._AUDIT_RETENTION_MIN_AML_DAYS} days (10 years) per "
                f"the platform's compliance hard-lock policy §M6 — got {self.AUDIT_RETENTION_DAYS_AML}."
            )


settings = Settings()
