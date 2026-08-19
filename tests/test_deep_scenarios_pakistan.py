"""
Pakistan pack — deep scenario tests (LOW + HIGH band per scenario).

Tests all 15 scenarios (13 conventional + 2 Islamic) against the live
Checkmarble instance. Each test class covers one scenario with two band checks:
  - test_low : payload designed to score in LOW band (1–24)
  - test_high : payload designed to score ≥ 25 (MEDIUM / HIGH / CRITICAL)

Run:
    set -a; source .env; set +a
    CHECKMARBLE_API_URL=http://localhost:8080 BRIDGE_URL=http://localhost:8000 \\
    FIREBASE_URL=https://identitytoolkit.googleapis.com FIREBASE_API_KEY="$FIREBASE_API_KEY" \\
    CREATE_ORG_ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD='Marble123!' \\
    python3 -m pytest tests/test_deep_scenarios_pakistan.py -v

Prerequisites:
    Pakistan pack must be loaded:
        ENABLE_ISLAMIC_SCENARIOS=true \\
        FIREBASE_URL=https://identitytoolkit.googleapis.com \\
        FIREBASE_API_KEY="$FIREBASE_API_KEY" \\
        ADMIN_PASSWORD='Marble123!' \\
        python3 scripts/rules_library/pakistan.py
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest

API_URL = os.getenv("CHECKMARBLE_API_URL", "http://localhost:8080")
API_KEY = os.getenv("CHECKMARBLE_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

_scenario_ids: dict[str, str] = {}


def _uid() -> str:
    return f"pk-{uuid.uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decide(scenario_name: str, object_type: str, trigger_object: dict) -> dict:
    sid = _scenario_ids.get(scenario_name)
    if not sid:
        pytest.skip(f"Scenario '{scenario_name}' not found in Marble — is Pakistan pack loaded?")
    trigger_object.setdefault("object_id", _uid())
    trigger_object.setdefault("updated_at", _now())
    r = httpx.post(f"{API_URL}/decisions", json={
        "scenario_id": sid,
        "object_type": object_type,
        "trigger_object": trigger_object,
    }, headers=HEADERS, timeout=30)
    assert r.status_code == 200, f"Decision failed ({r.status_code}): {r.text[:300]}"
    return r.json()


def _get_jwt() -> str:
    firebase_url = os.getenv("FIREBASE_URL", "http://localhost:9099")
    api_key      = os.getenv("FIREBASE_API_KEY", "placeholder")
    admin_email  = os.getenv("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
    admin_pass   = os.environ["ADMIN_PASSWORD"]
    if "localhost" in firebase_url or ":9099" in firebase_url:
        url = f"{firebase_url}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    else:
        url = f"{firebase_url}/v1/accounts:signInWithPassword?key={api_key}"
    r = httpx.post(url, json={"email": admin_email, "password": admin_pass, "returnSecureToken": True})
    id_token = r.json()["idToken"]
    r2 = httpx.post(f"{API_URL}/token", headers={"Authorization": f"Bearer {id_token}"})
    return r2.json()["access_token"]


@pytest.fixture(autouse=True, scope="session")
def load_scenario_ids():
    try:
        jwt = _get_jwt()
        r = httpx.get(f"{API_URL}/scenarios", headers={"Authorization": f"Bearer {jwt}"}, timeout=10)
        if r.status_code == 200:
            for s in r.json():
                _scenario_ids[s["name"]] = s["id"]
    except Exception as e:
        pytest.skip(f"Cannot connect to Marble API: {e}")


@pytest.fixture(autouse=True)
def check_api():
    try:
        r = httpx.get(f"{API_URL}/liveness", timeout=5)
        if r.status_code != 200:
            pytest.skip("Marble API not available")
    except httpx.ConnectError:
        pytest.skip("Marble API not reachable")


# ===========================================================================
# 1. KYC Verification Scoring  (individual_users)
# Rules: KYC_CNIC_001 (+25), KYC_NAME_002 (+20), KYC_AGE_003 (+15),
#        KYC_DOC_004 (+25), KYC_BIO_005 (+15)
# ===========================================================================
class TestKycVerificationScoring:
    def test_low(self):
        result = _decide("kyc_verification_scoring", "individual_users", {
            "user_uid": _uid(),
            "cnic_number": "42101-1234567-1",   # valid format
            "id_image_url": "https://cdn.example.com/id.jpg",
            "customer_age": 35,
            "biometric_registered": True,
            # FDS pre-computed
            "cnic_format_valid": True,
            "nadra_verisys_match": True,
            "ocr_passed": True,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("kyc_verification_scoring", "individual_users", {
            "user_uid": _uid(),
            "cnic_number": "INVALID-CNIC",       # KYC_CNIC_001 +25
            "id_image_url": "",                   # KYC_DOC_004 +25
            "customer_age": 35,
            "biometric_registered": False,        # KYC_BIO_005 +15
            # FDS pre-computed
            "cnic_format_valid": False,
            "nadra_verisys_match": False,         # KYC_NAME_002 +20
            "ocr_passed": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 2. User Onboarding Risk  (individual_users)
# Rules: OBR_PEP_010 (+20), OBR_SANC_011 (+20), OBR_GEO_012 (+10),
#        OBR_OCC_013 (+10), OBR_VEL_014 (+5)
# ===========================================================================
class TestUserOnboardingRisk:
    def test_low(self):
        result = _decide("user_onboarding_risk", "individual_users", {
            "user_uid": _uid(),
            "pep_match": False,
            "sanctions_hit": False,
            "home_high_risk_geo": False,
            "occupation_high_risk": False,
            "cnic_account_count_30d": 1,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("user_onboarding_risk", "individual_users", {
            "user_uid": _uid(),
            "pep_match": True,            # OBR_PEP_010 +20
            "sanctions_hit": True,        # OBR_SANC_011 +20
            "home_high_risk_geo": True,   # OBR_GEO_012 +10
            "occupation_high_risk": True, # OBR_OCC_013 +10
            "cnic_account_count_30d": 6,  # OBR_VEL_014 +5
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 3. Identifier / Format Validation  (individual_users or transactions)
# Rules: IBAN_PK_030 (+10), IBAN_LEN_031 (+5), MOB_PK_032 (+10),
#        CNIC_FMT_033 (+5), EMAIL_034 (+5)
# ===========================================================================
class TestIdentifierValidation:
    def test_low(self):
        result = _decide("kyc_verification_scoring", "individual_users", {
            "user_uid": _uid(),
            "cnic_number": "42101-1234567-1",
            "id_image_url": "https://cdn.example.com/id.jpg",  # non-empty → KYC_DOC_004 won't fire
            "customer_age": 30,
            "biometric_registered": True,
            "cnic_format_valid": True,
            "nadra_verisys_match": True,
            "ocr_passed": True,
            "iban_format_valid": True,
            "iban_length_24": True,
            "mobile_format_valid": True,
            "email_format_valid": True,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("kyc_verification_scoring", "individual_users", {
            "user_uid": _uid(),
            "cnic_number": "INVALID",
            "customer_age": 30,
            "biometric_registered": False,
            "cnic_format_valid": False,     # KYC_CNIC_001 +25
            "nadra_verisys_match": False,   # KYC_NAME_002 +20
            "ocr_passed": False,            # KYC_DOC_004 +25
            "iban_format_valid": False,
            "iban_length_24": False,
            "mobile_format_valid": False,
            "email_format_valid": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 4. AML Compliance Screening  (transactions)
# Rules: CMP_CTR_AMT (+20), CMP_CTR_AGG (+15), CMP_STR_LRG (+15),
#        CMP_STR_STR (+15), CMP_PEP_020 (+10), CMP_SANC_021 (+10),
#        CMP_HIGH_BIZ_022 (+10), CMP_WIRE_023 (+5)
# ===========================================================================
class TestAmlComplianceScreening:
    def test_low(self):
        result = _decide("aml_compliance_screening", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-pk-001",
            "to_wallet_uid": "merchant-pk-001",
            "amount": 5000,
            "currency": "PKR",
            "transaction_type": "transfer",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "cash_24h_sum_pkr": 10000.0,
            "withdrawal_1h_sum_pkr": 5000.0,
            "structuring_detected": False,
            "counterparty_pep": False,
            "counterparty_sanctions": False,
            "merchant_high_risk": False,
            "wire_originator_complete": True,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("aml_compliance_screening", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-pk-002",
            "to_wallet_uid": "merchant-pk-002",
            "amount": 3000000,
            "currency": "PKR",
            "transaction_type": "cash",        # CMP_CTR_AMT: cash >= 2M +20
            "transaction_status": "Pending",
            "timestamp": _now(),
            "cash_24h_sum_pkr": 2500000.0,     # CMP_CTR_AGG: >= 2M +15
            "withdrawal_1h_sum_pkr": 600000.0, # CMP_STR_LRG: >= 500k +15
            "structuring_detected": True,       # CMP_STR_STR +15
            "counterparty_pep": False,
            "counterparty_sanctions": False,
            "merchant_high_risk": False,
            "wire_originator_complete": True,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 5. Transaction Risk Scoring  (transactions)
# Rules: RSK_VEL_050 (+12), RSK_NIGHT_051 (+8), RSK_GEO_052 (+10),
#        RSK_GEO_MISMATCH_053 (+10), RSK_DEV_054 (+10), RSK_VPN_055 (+8),
#        RSK_DORM_056 (+8), RSK_DEPS_057 (+8), RSK_ROUND_058 (+5),
#        RSK_TIME_059 (+5), RSK_FCY_060 (+5), RSK_CB_061 (+10),
#        RSK_SHELL_062 (+5)
# ===========================================================================
class TestTransactionRiskScoring:
    def test_low(self):
        result = _decide("transaction_risk_scoring", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-low-001",
            "to_wallet_uid": "merchant-low-001",
            "amount": 2000,
            "currency": "PKR",
            "transaction_type": "transfer",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "velocity_spike_detected": False,
            "is_night_tx": False,
            "geo_high_risk": False,
            "home_vs_txn_province_mismatch": False,
            "imei_mismatch_flag": False,
            "vpn_usage": False,
            "dormant_reactivation_flag": False,
            "small_deposits_24h_count": 0,
            "round_amount_sub_ctr": False,
            "burst_pattern_detected": False,
            "counterparty_jurisdiction_fatf_listed": False,
            "counterparty_shell_pattern": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("transaction_risk_scoring", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-high-001",
            "to_wallet_uid": "merchant-high-001",
            "amount": 50000,
            "currency": "PKR",
            "transaction_type": "transfer",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "velocity_spike_detected": True,              # RSK_VEL_050 +12
            "is_night_tx": True,                          # RSK_NIGHT_051 +8
            "geo_high_risk": True,                        # RSK_GEO_052 +10
            "home_vs_txn_province_mismatch": True,        # RSK_GEO_MISMATCH_053 +10
            "imei_mismatch_flag": True,                   # RSK_DEV_054 +10
            "vpn_usage": False,
            "dormant_reactivation_flag": False,
            "small_deposits_24h_count": 0,
            "round_amount_sub_ctr": False,
            "burst_pattern_detected": False,
            "counterparty_jurisdiction_fatf_listed": False,
            "counterparty_shell_pattern": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 6. Biometric Fraud Detection  (biometric_logs)
# Rules: BIO_FAIL_070 (+25), BIO_LIVE_071 (+25), BIO_NEW_072 (+20),
#        BIO_NADRA_073 (+15), BIO_GEO_074 (+10), BIO_CHANNEL_075 (+5)
# ===========================================================================
class TestBiometricFraudDetection:
    def test_low(self):
        bid = _uid()
        result = _decide("biometric_fraud_detection", "biometric_logs", {
            "biometric_id": bid,
            "user_uid": _uid(),
            "biometric_type": "face",
            "retry_count": 1,                    # < 3, BIO_FAIL_070 doesn't fire
            "liveness_check_passed": True,
            "biometric_recent_reenroll": False,
            "nadra_verisys_match": True,
            "biometric_impossible_travel": False,
            "biometric_channel_violation": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        bid = _uid()
        result = _decide("biometric_fraud_detection", "biometric_logs", {
            "biometric_id": bid,
            "user_uid": _uid(),
            "biometric_type": "face",
            "retry_count": 5,                    # >= 3 → BIO_FAIL_070 +25
            "liveness_check_passed": False,       # BIO_LIVE_071 +25
            "biometric_recent_reenroll": False,
            "nadra_verisys_match": True,
            "biometric_impossible_travel": False,
            "biometric_channel_violation": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 7. Agent GPS / IMEI Compliance  (agents)
# Rules: AGT_LIC_080 (+25), AGT_GPS_081 (+20), AGT_IMEI_082 (+20),
#        FLT_CAP_083 (+15), AGT_HRS_084 (+10), AGT_VOL_085 (+10)
# ===========================================================================
class TestAgentGpsImeiCompliance:
    def test_low(self):
        result = _decide("agent_gps_imei_compliance", "agents", {
            "agent_uid": _uid(),
            "agent_name": "Good Agent",
            "agent_license_status": "active",
            "risk_limit_exceeded": False,
            "agent_gps_in_service_area": True,
            "agent_imei_registered": True,
            "agent_outside_op_hours": False,
            "agent_txn_count_1h": 5,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("agent_gps_imei_compliance", "agents", {
            "agent_uid": _uid(),
            "agent_name": "Risky Agent",
            "agent_license_status": "expired",   # AGT_LIC_080 +25
            "risk_limit_exceeded": False,
            "agent_gps_in_service_area": False,  # AGT_GPS_081 +20
            "agent_imei_registered": False,       # AGT_IMEI_082 +20
            "agent_outside_op_hours": False,
            "agent_txn_count_1h": 5,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 8. Agent Float Monitoring  (float_delegations)
# Rules: FLT_DEP_090 (+20), FLT_NIGHT_091 (+15), FLT_VEL_092 (+20),
#        FLT_CASH_093 (+20), FLT_TURN_094 (+15), FLT_PROV_095 (+10)
# ===========================================================================
class TestAgentFloatMonitoring:
    def test_low(self):
        result = _decide("agent_float_monitoring", "float_delegations", {
            "delegation_uid": _uid(),
            "agent_uid": _uid(),
            "agent_topup_over_cap": False,
            "float_depletion_pct_1h": 0.3,
            "float_turnover_ratio": 1.0,
            "agent_province_mismatch": False,
            "is_night_tx": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("agent_float_monitoring", "float_delegations", {
            "delegation_uid": _uid(),
            "agent_uid": _uid(),
            "agent_topup_over_cap": True,         # FLT_DEP_090 +20
            "float_depletion_pct_1h": 85.0,       # FLT_VEL_092 > 80.0 (%) +20
            "float_turnover_ratio": 1.0,
            "agent_province_mismatch": False,
            "is_night_tx": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 9. Merchant License Compliance  (merchant_users)
# Rules: MCH_LIC_100 (+30), MCH_CAT_101 (+25), MCH_VOL_102 (+20),
#        MCH_LOC_103 (+15), MCH_FBR_104 (+10)
# ===========================================================================
class TestMerchantLicenseCompliance:
    def test_low(self):
        mid = _uid()
        result = _decide("merchant_license_compliance", "merchant_users", {
            "merchant_uid": mid,
            "merchant_name": "Good Merchant PK",
            "merchant_license_status": "active",
            "merchant_category": "Retail",
            "fbr_ntn_registered": True,
            "merchant_high_risk": False,
            "merchant_sbp_approved": True,
            "merchant_volume_ratio_30d": 1.2,
            "merchant_location_mismatch": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        mid = _uid()
        result = _decide("merchant_license_compliance", "merchant_users", {
            "merchant_uid": mid,
            "merchant_name": "Risky Merchant PK",
            "merchant_license_status": "expired",  # MCH_LIC_100 +30
            "merchant_category": "Retail",
            "fbr_ntn_registered": False,            # MCH_FBR_104 +10
            "merchant_high_risk": True,             # MCH_CAT_101 +25
            "merchant_sbp_approved": False,
            "merchant_volume_ratio_30d": 1.2,
            "merchant_location_mismatch": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 10. Login Anomaly Detection  (individual_users)
# Rules: LOGIN_GEO_110 (+30), LOGIN_DEV_111 (+25), LOGIN_FAIL_112 (+25),
#        LOGIN_VPN_113 (+20)
# ===========================================================================
class TestLoginAnomalyDetection:
    def test_low(self):
        result = _decide("login_anomaly_detection", "individual_users", {
            "user_uid": _uid(),
            "login_country": "PK",
            "travel_notice_active": False,
            "login_new_device": False,
            "login_new_ip": False,
            "failed_logins_10min": 1,
            "vpn_usage": False,
            "session_high_risk": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("login_anomaly_detection", "individual_users", {
            "user_uid": _uid(),
            "login_country": "IN",          # != "PK" + no travel notice → LOGIN_GEO_110 +30
            "travel_notice_active": False,
            "login_new_device": True,       # + login_new_ip → LOGIN_DEV_111 +25
            "login_new_ip": True,
            "failed_logins_10min": 6,       # >= 5 → LOGIN_FAIL_112 +25
            "vpn_usage": False,
            "session_high_risk": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 11. Dormant Account Reactivation  (individual_users)
# Rules: RSK_DORM_056 (+8 in txn scoring) — dormant-specific:
#        DORM_AMT_120 (+40), DORM_KYC_121 (+30), DORM_PAT_122 (+30)
# ===========================================================================
class TestDormantAccountReactivation:
    def test_low(self):
        result = _decide("dormant_account_reactivation", "individual_users", {
            "user_uid": _uid(),
            "account_status": "Active",
            "dormant_reactivation_flag": False,
            "rekyc_completed": True,
            "dormant_outflow_pattern": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("dormant_account_reactivation", "individual_users", {
            "user_uid": _uid(),
            "account_status": "Dormant",
            "dormant_reactivation_flag": True,
            "rekyc_completed": False,       # DORM_KYC_121 +30
            "dormant_outflow_pattern": True, # DORM_PAT_122 +30
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 12. POS Transaction Compliance  (transactions)
# Rules: POS_MCC_130 (+40), POS_AMT_131 (+30), POS_INTL_132 (+30)
# ===========================================================================
class TestPosTransactionCompliance:
    def test_low(self):
        result = _decide("pos_transaction_compliance", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-pos-001",
            "to_wallet_uid": "merchant-pos-001",
            "amount": 5000,
            "currency": "PKR",
            "transaction_type": "pos",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "pos_mcc_restricted": False,
            "pos_international": False,
            "travel_notice_active": True,
            "amount_usd_equiv": 30.0,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("pos_transaction_compliance", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-pos-002",
            "to_wallet_uid": "merchant-pos-002",
            "amount": 600000,
            "currency": "PKR",
            "transaction_type": "pos",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "pos_mcc_restricted": True,      # POS_MCC_130 +40
            "pos_international": True,
            "travel_notice_active": False,   # + international + no notice → POS_INTL_132 +30
            "amount_usd_equiv": 600.0,       # POS_AMT_131: amount_usd_equiv >= 500 → +30
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 13. Notification Failure Monitoring  (notification_logs)
# Rules: NTF_OTP_140 (+25), NTF_SMS_141 (+20), NTF_EMAIL_142 (+20),
#        NTF_PUSH_143 (+20), NTF_RISK_144 (+15)
# ===========================================================================
class TestNotificationFailureMonitoring:
    def test_low(self):
        result = _decide("notification_failure_monitoring", "notification_logs", {
            "notification_id": _uid(),
            "user_uid": _uid(),
            "event_type": "balance_inquiry",
            "delivery_status": "Delivered",
            "otp_failures_1h": 0,
            "sms_failure_rate_1h": 0.5,
            "email_bounced": False,
            "all_notif_channels_failed": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("notification_failure_monitoring", "notification_logs", {
            "notification_id": _uid(),
            "user_uid": _uid(),
            "event_type": "txn_confirm",          # + delivery_status=Failed → NTF_PUSH_143 +20
            "delivery_status": "Failed",
            "otp_failures_1h": 5,                 # > 3 → NTF_OTP_140 +25
            "sms_failure_rate_1h": 6.0,           # > 5 → NTF_SMS_141 +20
            "email_bounced": False,
            "all_notif_channels_failed": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 14. Corporate Float Monitoring  (corporate_wallets)
# Rules: CORP_VOL_150 (+25), CORP_BENF_151 (+25), CORP_KYC_152 (+25),
#        CORP_INVOICE_153 (+15), CORP_SUBS_154 (+10)
# ===========================================================================
class TestCorporateFloatMonitoring:
    def test_low(self):
        cid = _uid()
        result = _decide("corporate_float_monitoring", "corporate_wallets", {
            "corporate_id": cid,
            "organization_name": "Good Corp PK",
            "amount": 100000,
            "beneficial_owner_declared": True,
            "corp_volume_ratio_30d": 1.5,       # < 3, no CORP_VOL
            "corp_kyc_stale_12m": False,
            "corp_invoice_mismatch": False,
            "counterparty_new_subsidiary": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        cid = _uid()
        result = _decide("corporate_float_monitoring", "corporate_wallets", {
            "corporate_id": cid,
            "organization_name": "Risky Corp PK",
            "amount": 6000000,                  # > 5M + no BO → CORP_BENF_151 +25
            "beneficial_owner_declared": False,
            "corp_volume_ratio_30d": 4.0,       # > 3 → CORP_VOL_150 +25
            "corp_kyc_stale_12m": True,          # CORP_KYC_152 +25
            "corp_invoice_mismatch": False,
            "counterparty_new_subsidiary": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 15. Shariah Contract Compliance  (transactions)  [Islamic pack]
# Rules: SHR_TYPE_040 (+30), SHR_RIBA_041 (+30), SHR_VIOL_042 (+25),
#        SHR_TT_043 (+15)
# ===========================================================================
class TestShariahContractCompliance:
    def test_low(self):
        result = _decide("shariah_contract_compliance", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-shr-001",
            "to_wallet_uid": "merchant-shr-001",
            "amount": 10000,
            "currency": "PKR",
            "transaction_type": "transfer",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "lms_shariah_tag": "MRB",           # valid contract tag — no SHR_TYPE
            "interest_rate": 0,                  # = 0 — no SHR_RIBA
            "shariah_violation_detected": False,
            "shariah_tt_code_mismatch": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("shariah_contract_compliance", "transactions", {
            "transaction_id": _uid(),
            "from_wallet_uid": "user-shr-002",
            "to_wallet_uid": "merchant-shr-002",
            "amount": 10000,
            "currency": "PKR",
            "transaction_type": "transfer",
            "transaction_status": "Pending",
            "timestamp": _now(),
            "lms_shariah_tag": "MRB",
            "interest_rate": 5,                  # > 0 on Islamic tag → SHR_RIBA_041 +30
            "shariah_violation_detected": True,  # SHR_VIOL_042 +25
            "shariah_tt_code_mismatch": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"


# ===========================================================================
# 16. Wakalah Delegation Compliance  (wakalah_delegations)  [Islamic pack]
# Rules: WKL_BLOCK_160 (+30), WKL_AMT_161 (+25), WKL_EXPIRY_162 (+25),
#        WKL_SCOPE_163 (+20)
# ===========================================================================
class TestWakalahDelegationCompliance:
    def test_low(self):
        result = _decide("wakalah_delegation_compliance", "wakalah_delegations", {
            "delegation_uid": _uid(),
            "principal_uid": _uid(),
            "wakil_uid": _uid(),
            "contract_type": "Wakalah",
            "amount": 50000,                 # < 1M — no WKL_AMT
            "wakalah_revoked": False,
            "wakalah_expired": False,
            "wakalah_out_of_scope": False,
        })
        score = result["score"]
        assert 0 <= score < 25, f"Expected LOW score (0-24), got {score}"

    def test_high(self):
        result = _decide("wakalah_delegation_compliance", "wakalah_delegations", {
            "delegation_uid": _uid(),
            "principal_uid": _uid(),
            "wakil_uid": _uid(),
            "contract_type": "Wakalah",
            "amount": 1500000,               # > 1M Wakalah → WKL_AMT_161 +25
            "wakalah_revoked": True,          # WKL_BLOCK_160 +30
            "wakalah_expired": True,          # WKL_EXPIRY_162 +25
            "wakalah_out_of_scope": False,
        })
        score = result["score"]
        assert score >= 25, f"Expected HIGH score >=25, got {score}"
