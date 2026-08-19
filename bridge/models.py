"""Pydantic models for all 12 data tables, decisions, and webhooks.

Canonical references:
    - spec.md §3 (data model), §7.4 (canonical output payload)
    - docs/jurisdictions/pakistan/REGULATORY_MAPPING.md (reference pack field mapping)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Canonical enums for the output contract (spec.md §7.4, GNN PDF §2.1)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Canonical 4-tier risk classification.

    Mapped from combined_score:
        0-24  → Low
        25-39 → Medium
        40-59 → High
        60+   → Critical
    """
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class FinalDecisionSuggestion(str, Enum):
    """Canonical decision suggestion (GNN PDF §2.1).

    Mapped from combined_score + context:
        0-24         → Allow     (pass through)
        25-39        → Monitor   (log + notify, no block)
        40-59        → Escalate  (to compliance officer)
        60+          → Escalate  (with STR + auto-hold)
        any + review → SoftHold  (conditional hold pending review)
    """
    ALLOW = "Allow"
    MONITOR = "Monitor"
    ESCALATE = "Escalate"
    SOFT_HOLD = "SoftHold"


class ComplianceViolationType(str, Enum):
    """Primary compliance-violation category (GNN PDF §2.1).

    Derived from the dominant triggered-rule category. Used by compliance
    queue routing to assign cases to the right officer specialization.
    """
    AML = "AML"
    STR = "STR"
    CTR = "CTR"
    SHARIAH = "Shariah"
    VELOCITY = "Velocity"
    DEVICE = "Device"
    SANCTIONS = "Sanctions"
    NONE = "None"  # no specific violation (score below threshold)


# --- Data Table Models (12 tables) ---

class IndividualUser(BaseModel):
    user_id: str
    full_name: str = ""
    national_id_number: str = ""
    date_of_birth: Optional[datetime] = None
    gender: str = ""
    mobile_number: str = ""
    email_address: str = ""
    address_province: str = ""
    address_district: str = ""
    address_full: str = ""
    face_biometric_registered: bool = False
    voice_biometric_registered: bool = False
    biometric_retry_count: int = 0
    last_login_imei: str = ""
    device_os: str = ""
    account_status: str = "Active"
    otp_enabled: bool = False
    kyc_level: str = "L1"
    kyc_submission_date: Optional[datetime] = None
    kyc_status: str = "pending"
    biometric_delete_attempts: int = 0
    blacklisted_flag: bool = False
    last_transaction_amount: float = 0
    last_transaction_date: Optional[datetime] = None
    transaction_note: str = ""
    ip_address_last_login: str = ""
    geo_location_last_login: str = ""
    account_login_attempts_24h: int = 0
    pin_set: bool = False
    face_biometric_verified: bool = False
    voice_biometric_verified: bool = False
    l2_upgrade_attempts: int = 0
    l2_upgrade_status: str = ""
    tin_number: str = ""
    l2_kyc_documents_submitted: bool = False
    l2_kyc_approval_date: Optional[datetime] = None
    biometric_update_date: Optional[datetime] = None
    multi_device_login_flag: bool = False
    risk_score: float = 0
    last_otp_sent_date: Optional[datetime] = None


class MerchantUser(BaseModel):
    merchant_uid: str
    merchant_name: str = ""
    merchant_tin: str = ""
    business_license_no: str = ""
    license_expiry_date: Optional[datetime] = None
    kyc_level: str = "L3"
    contact_number: str = ""
    email: str = ""
    business_category: str = ""
    province_code: str = ""
    district_code: str = ""
    merchant_iban: str = ""
    registration_source: str = ""
    registration_date: Optional[datetime] = None
    status: str = "Active"
    last_activity: Optional[datetime] = None
    aml_flag: bool = False
    risk_score: float = 0
    pos_enabled: bool = False
    agent_referral_id: str = ""


class CorporateWallet(BaseModel):
    corporate_id: str
    organization_name: str = ""
    org_type: str = ""
    tin_number: str = ""
    business_license_number: str = ""
    registration_date: Optional[datetime] = None
    license_expiry_date: Optional[datetime] = None
    head_office_address: str = ""
    province_code: str = ""
    district_code: str = ""
    authorized_signatory_name: str = ""
    signatory_position: str = ""
    signatory_nid: str = ""
    signatory_contact: str = ""
    multi_user_enabled: bool = False
    approval_matrix_uploaded: bool = False
    bank_account_linked: bool = False
    iban_number: str = ""
    wallet_id: str = ""
    assigned_float_limit: float = 0
    used_float_today: float = 0
    available_float: float = 0
    risk_score: float = 0
    compliance_flag: str = "clear"
    staff_kyc_status: str = ""
    last_transaction_date: Optional[datetime] = None
    gps_location: str = ""
    imei_device_id: str = ""
    otp_enabled: bool = False
    voice_auth_enabled: bool = False
    biometric_auth_enabled: bool = False
    document_upload_status: bool = False
    contract_type: str = ""
    shariah_model: str = ""
    compliance_verified: bool = False
    audit_logs_enabled: bool = True


class Agent(BaseModel):
    agent_id: str
    full_name: str = ""
    phone_number: str = ""
    email: str = ""
    gender: str = ""
    date_of_birth: Optional[datetime] = None
    national_id_number: str = ""
    national_id_issue_date: Optional[datetime] = None
    national_id_expiry_date: Optional[datetime] = None
    province_code: int = 0
    district_code: int = 0
    village_or_zone: str = ""
    agent_role: str = ""
    license_type: str = ""
    license_number: str = ""
    license_expiry_date: Optional[datetime] = None
    referring_super_agent: str = ""
    imei_number: str = ""
    device_model: str = ""
    gps_lat: float = 0
    gps_long: float = 0
    biometric_status: str = ""
    voice_biometric: bool = False
    face_biometric: bool = False
    kyc_level: str = "L1"
    float_limit_afn: float = 0
    float_used_afn: float = 0
    risk_score: float = 0
    status: str = "Active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    google_auth_enabled: bool = False
    wallet_linked: bool = False
    iban_number: str = ""
    uid_code: str = ""
    contract_type: str = ""
    shariah_model: str = ""
    audit_flag: bool = False
    account_status: str = "Active"


class WakalahDelegation(BaseModel):
    wakalah_id: str
    principal_uid: str = ""
    wakil_uid: str = ""
    delegation_scope: str = ""
    permissions_granted: str = ""
    contract_type: str = ""
    contract_document_url: str = ""
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    revoked: bool = False
    revoked_by: str = ""
    revoked_at: Optional[datetime] = None
    audit_flag: bool = False
    last_used_for_txn: Optional[datetime] = None
    notes: str = ""


class Transaction(BaseModel):
    transaction_id: str
    from_wallet_uid: str = ""
    to_wallet_uid: str = ""
    amount: float = 0
    currency: str = "AFN"
    transaction_type: str = ""
    transaction_status: str = ""
    risk_score: float = 0
    timestamp: Optional[datetime] = None
    geo_location: str = ""
    imei: str = ""
    channel: str = ""
    initiated_by: str = ""
    approved_by: str = ""
    approved_at: Optional[datetime] = None
    reversal_flag: bool = False
    reversal_reason: str = ""
    reversal_txn_id: str = ""
    notes: str = ""
    audit_flag: bool = False


class NotificationLog(BaseModel):
    notification_id: str
    wallet_uid: str = ""
    user_type: str = ""
    notification_type: str = ""
    delivery_channel: str = ""
    message_title: str = ""
    message_body: str = ""
    sent_timestamp: Optional[datetime] = None
    delivery_status: str = ""
    retry_count: int = 0
    read_status: bool = False
    read_timestamp: Optional[datetime] = None
    fallback_used: bool = False
    fallback_method: str = ""
    event_triggered_by: str = ""
    event_type: str = ""
    audit_flag: bool = False


class RiskEvent(BaseModel):
    risk_id: str
    wallet_uid: str = ""
    agent_id: str = ""
    risk_score: float = 0
    risk_category: str = "low"
    trigger_type: str = ""
    trigger_description: str = ""
    gps_location: str = ""
    imei: str = ""
    velocity_count: int = 0
    timestamp: Optional[datetime] = None
    behavior_flag: bool = False
    device_mismatch: bool = False
    geo_mismatch: bool = False
    kyc_retries: int = 0
    float_usage_pct: float = 0
    transaction_volume: float = 0
    alert_status: str = "active"
    review_status: str = "pending"
    compliance_action: str = ""
    str_triggered: bool = False
    ctr_triggered: bool = False
    notes: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AuditTrail(BaseModel):
    audit_id: str
    action_type: str = ""
    performed_by_uid: str = ""
    performed_by_role: str = ""
    target_uid: str = ""
    target_entity_type: str = ""
    action_description: str = ""
    timestamp: Optional[datetime] = None
    ip_address: str = ""
    device_imei: str = ""
    geo_location: str = ""
    status_before: str = ""
    status_after: str = ""
    linked_risk_event: str = ""
    audit_verified: bool = False


class BiometricLog(BaseModel):
    biometric_id: str
    wallet_uid: str = ""
    user_type: str = ""
    biometric_type: str = ""
    enrollment_status: bool = False
    last_verified: Optional[datetime] = None
    verification_score: float = 0
    device_imei: str = ""
    geo_location: str = ""
    encryption_hash: str = ""
    retry_count: int = 0
    fallback_used: str = ""
    linked_kyc_level: str = ""
    voice_sample_file: str = ""
    face_image_file: str = ""
    offline_enrollment: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FloatDelegation(BaseModel):
    delegation_id: str
    delegated_from_uid: str = ""
    delegated_to_uid: str = ""
    delegation_type: str = ""
    delegated_amount: float = 0
    currency: str = "AFN"
    gps_location: str = ""
    imei: str = ""
    status: str = "Active"
    delegated_by: str = ""
    delegated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    float_used_today: float = 0
    float_balance: float = 0
    risk_limit_exceeded: bool = False
    audit_flag: bool = False
    contract_type: str = ""
    shariah_model: str = ""


class FdsInputFeatures(BaseModel):
    """Pre-computed FDS input features per transaction.

    Backend is expected to populate these BEFORE calling the Checkmarble
    decision API so that time-windowed + aggregation-based rules can be
    evaluated without requiring SUM pivots (not supported in Checkmarble
    v0.59). Canonical source: Data Contract v2 §1 + GNN PDF §1.1.
    """

    transaction_id: str
    transaction_amount: float = 0
    transaction_type: str = ""  # IBFT/POS/QR/APS/WALLET
    transaction_purpose: str = ""

    # Velocity / burst
    biometric_failure_count: int = 0
    velocity_spike_detected: bool = False        # RSK_VEL_060: >5 txn/60s
    velocity_1h_count: int = 0                   # CMP_CTR_021: >=10 transfers/1h
    withdrawal_1h_sum_afn: float = 0             # CMP_STR_020: >100K AFN/1h
    volume_burst_detected: bool = False

    # Geo / device
    geo_shift_detected: bool = False
    geo_shift_distance_km: float = 0             # RSK_LOC_061, CMP_GPS_ANOMALY_024
    device_trust_score: float = 1.0              # 0-1
    vpn_usage: bool = False
    multi_device_login_flag: bool = False
    imei_mismatch_flag: bool = False             # RSK_DEV_063

    # Time
    is_night_tx: bool = False                    # RSK_TME_062: 23:00-05:00
    is_odd_hour_tx: bool = False                 # Risk Scoring.pdf: 00:00-03:00
    tx_hour_utc: int = 0

    # Auth abuse
    pin_reset_count_24h: int = 0
    otp_request_count_24h: int = 0

    # Identifiers / dormancy
    iban_mismatch_flag: bool = False
    merchant_license_expired_flag: bool = False
    account_dormancy_flag: bool = False

    # Graph / beneficiary / compliance (Data Contract §1)
    staff_biometric_mismatch_count: int = 0
    linked_wallets_count: int = 0
    repeat_beneficiary_flag: bool = False
    str_ctr_history_count: int = 0
    sanctions_hit: bool = False
    blacklisted_flag_current: bool = False
    str_history_flag: bool = False
    ocr_verified: bool = True
    referral_abuse_flag: bool = False
    manual_risk_flag: bool = False

    # Agent / float
    float_used_over_daily_avg_pct: float = 0     # RSK_FLT_065: >40% → +18
    float_usage_pct: float = 0                   # current / limit

    # Shariah (Data Contract §1 + GNN PDF §1.1)
    zakat_qr_violation_flag: bool = False
    product_prohibited_tag_flag: bool = False
    lms_shariah_tag: str = ""                    # MUR/MUD/MSH/IJAR/WAK/QH/SALAM/ISTISNA
    shariah_violation_detected: bool = False

    # TT + APS (GNN PDF §1.1)
    tt_code: int = 0                             # 10-77
    role_tier: str = ""                          # L1-L6


# --- Decision & Webhook Models ---

class RuleTriggered(BaseModel):
    rule_id: str = ""
    score: float = 0
    description: str = ""


class CheckmarbleDecision(BaseModel):
    decision_id: str = ""
    scenario_id: str = ""
    outcome: str = ""  # approve, review, block_and_review, decline
    score: float = 0
    rules_triggered: list[RuleTriggered] = Field(default_factory=list)
    trigger_object_type: str = ""
    trigger_object_id: str = ""


class WebhookEvent(BaseModel):
    event: str = ""  # decision.created, decision.updated, decision.escalated, list.updated
    decision_id: str = ""
    outcome: str = ""
    score: float = 0
    reviewed_by: str = ""
    notes: str = ""


class EscalationResult(BaseModel):
    """Canonical FDS output payload — spec.md §7.4 + Data Contract §2 + GNN PDF §2.1.

    Populated by ``bridge/escalation.py`` during ``escalate()`` and enriched
    by ``bridge/compliance.py`` with STR/CTR/Shariah/sanctions signals.
    """

    # --- Scoring (layers) ---
    checkmarble_score: float = 0
    marbel_score: Optional[float] = None
    gnn_score: Optional[float] = None
    combined_score: float = 0              # alias for risk_score; see below
    risk_score: float = 0                  # canonical name (0-100)
    risk_level: RiskLevel = RiskLevel.LOW  # Low / Medium / High / Critical

    # --- Which layers fired ---
    marbel_triggered: bool = False
    gnn_triggered: bool = False

    # --- Explainability ---
    feature_contributions: dict = Field(default_factory=dict)
    entity_graph_signals: dict = Field(default_factory=dict)
    triggered_rules: list[str] = Field(default_factory=list)  # list of rule_ids

    # --- Decision + ops ---
    final_decision_suggestion: FinalDecisionSuggestion = FinalDecisionSuggestion.ALLOW
    # the operator operational event (the platform's compliance hard-lock policy §4.1, §M1) — replaces
    # Marble's auto-block default with an event that explicitly requires
    # human Compliance Officer action. Populated from combined_score.
    operational_event: str = "PASS_THROUGH"
    customer_status_string: str = ""          # one of the 5 §4.2 approved strings
    latency_ms: int = 0                       # total /decide wall-clock latency
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit_log_reference: str = ""             # audit_trail.audit_id linkage
    escalation_path_id: str = ""              # UUID for this specific escalation chain

    # --- Compliance flags ---
    aml_risk_flag: bool = False
    str_suggested: bool = False
    ctr_required: bool = False
    sanctions_match: bool = False
    compliance_violation_type: ComplianceViolationType = ComplianceViolationType.NONE
    compliance_alert_reason: str = ""         # human-readable summary

    # --- Shariah ---
    shariah_violation_flag: bool = False
    shariah_anomaly_reason: str = ""

    # --- Case routing (filled by compliance queue manager; null until implemented) ---
    assigned_compliance_officer_id: Optional[str] = None


class IngestRequest(BaseModel):
    object_id: str
    updated_at: Optional[datetime] = None
    data: dict = Field(default_factory=dict)


class DecideRequest(BaseModel):
    scenario_id: str
    trigger_object_type: str
    trigger_object: dict = Field(
        ...,
        description="Full trigger object with all fields for rule evaluation. "
        "Must include 'object_id' and 'updated_at'.",
    )
