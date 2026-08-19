#!/usr/bin/env python3
"""
the operator Pakistan — Jurisdiction Rule Pack (Template)
========================================================

~85 SBP / FMU / AML Act 2010 / FATF compliance rules for a generic Pakistani
bank or EMI deployment. This is a **regulatory-floor template**: each
threshold and rule maps to a specific clause in a public regulation, but
deploying banks are expected to tighten thresholds and add bank-specific
rules per their internal risk policy.

Sources (all public — no bank consultation):
  - SBP AML/CFT/CPF Regulations (Revised)            sbp.org.pk/l_frame/Revised-AML-CFT-Regulations.pdf
  - SBP Branchless Banking Regulations               sbp.org.pk/bprd/2019/C10-Branchless-Banking-Regulations.pdf
  - SBP Foreign Exchange Manual                       sbp.org.pk/fe_manual/index.htm
  - Anti-Money Laundering Act 2010                    fmu.gov.pk
  - FMU goAML guides + typology reports               fmu.gov.pk
  - FATF Pakistan MER + 2022 follow-up                fatf-gafi.org/en/countries/detail/Pakistan.html
  - SBP BPD Circular 26/2005 (dormancy)               sbp.org.pk/bpd/2005/C26.htm

Citations in: docs/jurisdictions/pakistan/REGULATORY_MAPPING.md

Islamic-banking scenarios (shariah_contract_compliance, wakalah_delegation_compliance)
are DISABLED by default. Pakistan has both conventional and Islamic banks; this
pack defaults to the conventional case. Set ENABLE_ISLAMIC_SCENARIOS=true to
enable the 8 shariah/wakalah rules.

Prerequisites:
  - Platform provisioned: scripts/setup.py has been run for the target organization
  - Environment: CHECKMARBLE_API_KEY, CHECKMARBLE_API_URL (or defaults)
  - For real Firebase auth: FIREBASE_API_KEY, FIREBASE_URL=https://identitytoolkit.googleapis.com
  - For emulator auth: FIREBASE_URL=http://localhost:9099 (default)
  - ADMIN_PASSWORD for the org admin user
  - Python 3.10+

Usage:
  ADMIN_PASSWORD='Marble123!' python3 scripts/rules_library/pakistan.py

  # To also enable Islamic-banking scenarios:
  ENABLE_ISLAMIC_SCENARIOS=true ADMIN_PASSWORD='...' python3 scripts/rules_library/pakistan.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Platform-shared AST helpers — speak the same dialect as scripts/setup.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _ast_helpers import (
    payload, fds, const,
    gt, gte, lt, lte, eq, neq,
    and_, or_, not_,
    is_empty, is_not_empty,
    string_contains,
    add, sub, mul, div,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PACK_NAME = "pakistan"
API_URL = os.environ.get("CHECKMARBLE_API_URL", "http://localhost:8080")
FIREBASE_URL = os.environ.get("FIREBASE_URL", "http://localhost:9099")
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "placeholder")
ADMIN_EMAIL = os.environ.get("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    sys.exit("ERROR: ADMIN_PASSWORD env var is required.")

ENABLE_ISLAMIC_SCENARIOS = os.environ.get("ENABLE_ISLAMIC_SCENARIOS", "false").lower() == "true"

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "seed_data" / PACK_NAME

# Risk band thresholds (canonical 4-tier — same as Afghan pack)
THRESHOLDS = {
    "score_review_threshold": 25,
    "score_block_and_review_threshold": 40,
    "score_decline_threshold": 60,
}

# Maps list_name -> CSV file in seed_data/pakistan/
SEED_FILES = {
    "aml_blacklist": "aml_blacklist.csv",
    "sanctions_watchlist": "sanctions_watchlist.csv",
    "internal_blacklist": "internal_blacklist.csv",
    "high_risk_provinces": "high_risk_provinces.csv",
    "high_risk_business_categories": "high_risk_business_categories.csv",
    "shariah_contract_types": "shariah_contract_types.csv",
    "valid_province_codes": "valid_province_codes.csv",
    "valid_district_codes": "valid_district_codes.csv",
    "known_vpn_ips": "known_vpn_ips.csv",
    "blocked_imeis": "blocked_imeis.csv",
    "str_flagged_users": "str_flagged_users.csv",
    "agent_float_limits": "agent_float_limits.csv",
}


# ---------------------------------------------------------------------------
# RULES_BY_SCENARIO
#
# ~80 rules keyed by canonical scenario name. Each entry is a tuple:
#     (rule_id: str, score_modifier: int, ast_condition: dict)
#
# Rule IDs trace back to REGULATORY_MAPPING.md §3 (Pakistan canonical doc).
# Where the AST has no regex primitive, format-validation rules rely on
# pre-computed FDS flags from the bridge feature extractor (e.g.,
# fds("cnic_format_valid")) or fall back to is_empty / value-comparison checks.
# ---------------------------------------------------------------------------
RULES_BY_SCENARIO: dict[str, list[tuple[str, int, dict]]] = {

    # 3.1 KYC verification scoring — 5 rules, total 100
    "kyc_verification_scoring": [
        # KYC_CNIC_001: NADRA CNIC format invalid (XXXXX-XXXXXXX-X). Bridge
        # pre-computes cnic_format_valid; falls back to is_empty for missing field.
        ("KYC_CNIC_001_invalid_format", 25,
         or_(is_empty(payload("cnic_number")),
             eq(fds("cnic_format_valid"), const(False)))),
        # KYC_NAME_002: Name on file doesn't match NADRA Verisys (S12).
        ("KYC_NAME_002_nadra_mismatch", 20,
         eq(fds("nadra_verisys_match"), const(False))),
        # KYC_AGE_003: Underage (<18) or extreme-elderly (>80) trigger.
        ("KYC_AGE_003_age_out_of_band", 15,
         or_(lt(payload("customer_age"), const(18)),
             gt(payload("customer_age"), const(80)))),
        # KYC_DOC_004: ID image missing OR OCR failed (SBP AML/CFT Reg 1 §3.2).
        ("KYC_DOC_004_doc_missing_or_ocr_fail", 25,
         or_(is_empty(payload("id_image_url")),
             eq(fds("ocr_passed"), const(False)))),
        # KYC_BIO_005: Biometric template not registered (NADRA req. for Asaan).
        ("KYC_BIO_005_biometric_not_registered", 15,
         eq(payload("biometric_registered"), const(False))),
    ],

    # 3.2 + 3.4 User onboarding risk + identity format (10 rules, total 100)
    "user_onboarding_risk": [
        # OBR_PEP_010: Customer name matches PEP list (SBP AML/CFT Reg 1 §29).
        ("OBR_PEP_010_pep_match", 20,
         eq(fds("pep_match"), const(True))),
        # OBR_SANC_011: Customer matches sanctions watchlist (AML Act §9).
        ("OBR_SANC_011_sanctions_hit", 20,
         eq(fds("sanctions_hit"), const(True))),
        # OBR_GEO_012: Home district in critical-risk geo (FATF MER).
        ("OBR_GEO_012_home_critical_geo", 10,
         eq(fds("home_high_risk_geo"), const(True))),
        # OBR_OCC_013: Occupation in high-risk business category (S1 + S8).
        ("OBR_OCC_013_high_risk_occupation", 10,
         eq(fds("occupation_high_risk"), const(True))),
        # OBR_VEL_014: 3+ accounts opened by same CNIC in 30 days (S8 typology).
        ("OBR_VEL_014_multi_account_velocity", 5,
         gte(fds("cnic_account_count_30d"), const(3))),
        # IBAN_PK_030: IBAN format invalid (PK + 2 check + 4 bank + 16 account).
        ("IBAN_PK_030_iban_format_invalid", 10,
         or_(is_empty(payload("iban")),
             eq(fds("iban_format_valid"), const(False)))),
        # IBAN_LEN_031: IBAN length != 24.
        ("IBAN_LEN_031_iban_wrong_length", 5,
         eq(fds("iban_length_24"), const(False))),
        # MOB_PK_032: Mobile not in Pakistan format (+92 3XX XXXXXXX).
        ("MOB_PK_032_mobile_format_invalid", 10,
         eq(fds("mobile_format_valid"), const(False))),
        # CNIC_FMT_033: Reuses KYC_CNIC_001 logic at txn time.
        ("CNIC_FMT_033_txn_cnic_invalid", 5,
         eq(fds("cnic_format_valid"), const(False))),
        # EMAIL_034: Soft-flag if email format invalid (warning, not block).
        ("EMAIL_034_email_invalid", 5,
         eq(fds("email_format_valid"), const(False))),
    ],

    # 3.3 AML compliance screening — 8 rules, total 100
    "aml_compliance_screening": [
        # CMP_CTR_AMT: Single cash txn ≥ PKR 2M → CTR (AML Act §7; AML/CFT Reg 4 §3).
        ("CMP_CTR_AMT_single_2M", 20,
         and_(eq(payload("transaction_type"), const("cash")),
              gte(payload("amount"), const(2000000)))),
        # CMP_CTR_AGG: 24h aggregate cash deposits ≥ PKR 2M → CTR.
        ("CMP_CTR_AGG_24h_2M", 15,
         gte(fds("cash_24h_sum_pkr"), const(2000000.0))),
        # CMP_STR_LRG: Withdrawal ≥ PKR 500k in 1h (bank-tunable).
        ("CMP_STR_LRG_withdrawal_500k_1h", 15,
         gte(fds("withdrawal_1h_sum_pkr"), const(500000.0))),
        # CMP_STR_STR: Structuring/smurfing — 5+ deposits below CTR threshold in 24h.
        ("CMP_STR_STR_structuring", 15,
         eq(fds("structuring_detected"), const(True))),
        # CMP_PEP_020: Counterparty matches PEP list.
        ("CMP_PEP_020_counterparty_pep", 10,
         eq(fds("counterparty_pep"), const(True))),
        # CMP_SANC_021: Counterparty matches sanctions list.
        ("CMP_SANC_021_counterparty_sanctions", 10,
         eq(fds("counterparty_sanctions"), const(True))),
        # CMP_HIGH_BIZ_022: Merchant in high-risk business category.
        ("CMP_HIGH_BIZ_022_merchant_high_risk", 10,
         eq(fds("merchant_high_risk"), const(True))),
        # CMP_WIRE_023: Cross-border wire missing originator info (SBP requires
        # originator info on ALL wires, regardless of threshold — AML/CFT Reg 3).
        ("CMP_WIRE_023_originator_missing", 5,
         and_(eq(payload("transaction_type"), const("wire_transfer_crossborder")),
              eq(fds("wire_originator_complete"), const(False)))),
    ],

    # 3.6 Risk scoring & behavior — 13 rules, total 100
    "transaction_risk_scoring": [
        # RSK_VEL_050: Velocity spike (5+ txns in 60s) — pre-computed by bridge.
        ("RSK_VEL_050_velocity_spike", 12,
         eq(fds("velocity_spike_detected"), const(True))),
        # RSK_NIGHT_051: Night txn (00:00-05:00) AND amount > PKR 100k.
        ("RSK_NIGHT_051_night_high_value", 8,
         and_(eq(fds("is_night_tx"), const(True)),
              gt(payload("amount"), const(100000)))),
        # RSK_GEO_052: Txn geo in high-risk province polygon.
        ("RSK_GEO_052_high_risk_geo", 10,
         eq(fds("geo_high_risk"), const(True))),
        # RSK_GEO_MISMATCH_053: Home province != txn province AND no travel notice.
        ("RSK_GEO_MISMATCH_053_geo_mismatch", 10,
         and_(eq(fds("home_vs_txn_province_mismatch"), const(True)),
              eq(fds("travel_notice_active"), const(False)))),
        # RSK_DEV_054: IMEI mismatch from last login.
        ("RSK_DEV_054_imei_mismatch", 10,
         eq(fds("imei_mismatch_flag"), const(True))),
        # RSK_VPN_055: Source IP in known_vpn_ips.csv.
        ("RSK_VPN_055_vpn_in_use", 8,
         eq(fds("vpn_usage"), const(True))),
        # RSK_DORM_056: Dormant account suddenly active (S8 + dormancy reg).
        ("RSK_DORM_056_dormant_reactivated", 8,
         eq(fds("dormant_reactivation_flag"), const(True))),
        # RSK_DEPS_057: 10+ small deposits in 24h (smurfing pattern).
        ("RSK_DEPS_057_many_small_deposits", 8,
         gte(fds("small_deposits_24h_count"), const(10))),
        # RSK_ROUND_058: Suspiciously round amounts just below CTR (e.g., 1.99M).
        ("RSK_ROUND_058_sub_ctr_round", 5,
         eq(fds("round_amount_sub_ctr"), const(True))),
        # RSK_TIME_059: Txns clustered in <60s bursts.
        ("RSK_TIME_059_burst_pattern", 5,
         eq(fds("burst_pattern_detected"), const(True))),
        # RSK_FCY_060: FCY transaction → score boost (heightened scrutiny).
        ("RSK_FCY_060_fcy_txn", 5,
         and_(neq(payload("currency"), const("PKR")),
              is_not_empty(payload("currency")))),
        # RSK_CB_061: Cross-border to FATF grey/black-listed jurisdiction.
        ("RSK_CB_061_fatf_listed_jurisdiction", 10,
         eq(fds("counterparty_jurisdiction_fatf_listed"), const(True))),
        # RSK_SHELL_062: Transfer to known shell-company pattern.
        ("RSK_SHELL_062_shell_pattern", 5,
         eq(fds("counterparty_shell_pattern"), const(True))),
    ],

    # 3.7 Biometric fraud detection — 6 rules, total 100
    "biometric_fraud_detection": [
        # BIO_FAIL_070: 3+ biometric failures in 1h.
        # Uses biometric_logs.retry_count (Int) — Marble v0.59 cannot
        # delete/modify schema fields once created, and an earlier seed
        # added biometric_failure_count as Bool to this table; retry_count
        # is the existing Int counter on the same table and semantically
        # equivalent (failed biometric attempts in the session).
        ("BIO_FAIL_070_multi_failures", 25,
         gte(fds("retry_count"), const(3))),
        # BIO_LIVE_071: Liveness check failed.
        ("BIO_LIVE_071_liveness_failed", 25,
         eq(fds("liveness_check_passed"), const(False))),
        # BIO_NEW_072: Biometric template re-enrolled within 7 days of previous.
        ("BIO_NEW_072_recent_reenroll", 20,
         eq(fds("biometric_recent_reenroll"), const(True))),
        # BIO_NADRA_073: NADRA Verisys returned mismatch.
        ("BIO_NADRA_073_nadra_mismatch", 15,
         eq(fds("nadra_verisys_match"), const(False))),
        # BIO_GEO_074: Impossible-travel — biometric capture > 100km from last
        # within 1h.
        ("BIO_GEO_074_impossible_travel", 10,
         eq(fds("biometric_impossible_travel"), const(True))),
        # BIO_CHANNEL_075: Biometric used on unauthorized channel.
        ("BIO_CHANNEL_075_wrong_channel", 5,
         eq(fds("biometric_channel_violation"), const(True))),
    ],

    # 3.8 Agent + GPS + IMEI compliance — 6 rules, total 100
    "agent_gps_imei_compliance": [
        # AGT_LIC_080: Agent license expired or inactive.
        ("AGT_LIC_080_license_expired", 25,
         neq(payload("agent_license_status"), const("active"))),
        # AGT_GPS_081: Agent's GPS coords outside registered service area.
        ("AGT_GPS_081_outside_service_area", 20,
         eq(fds("agent_gps_in_service_area"), const(False))),
        # AGT_IMEI_082: Agent's IMEI changed without re-registration.
        ("AGT_IMEI_082_imei_unregistered", 20,
         eq(fds("agent_imei_registered"), const(False))),
        # FLT_CAP_083: Agent float exceeds tier cap from agent_float_limits.csv.
        ("FLT_CAP_083_tier_cap_exceeded", 15,
         eq(payload("risk_limit_exceeded"), const(True))),
        # AGT_HRS_084: Txns by agent outside registered operating hours.
        ("AGT_HRS_084_outside_op_hours", 10,
         eq(fds("agent_outside_op_hours"), const(True))),
        # AGT_VOL_085: Agent doing > 50 txns/hour (potential mule).
        ("AGT_VOL_085_high_volume_agent", 10,
         gt(fds("agent_txn_count_1h"), const(50))),
    ],

    # 3.9 Agent float monitoring — 6 rules, total 100
    "agent_float_monitoring": [
        # FLT_DEP_090: Agent float top-up > tier cap.
        ("FLT_DEP_090_topup_over_cap", 20,
         eq(fds("agent_topup_over_cap"), const(True))),
        # FLT_NIGHT_091: Float movements 00:00-05:00.
        ("FLT_NIGHT_091_night_float", 15,
         and_(eq(fds("is_night_tx"), const(True)),
              eq(payload("transaction_type"), const("agent_float")))),
        # FLT_VEL_092: Float depletion > 80% in 1 hour.
        ("FLT_VEL_092_depletion_80pct", 20,
         gt(fds("float_depletion_pct_1h"), const(80.0))),
        # FLT_CASH_093: Agent cashout > PKR 200k single txn.
        ("FLT_CASH_093_high_cashout", 20,
         and_(eq(payload("transaction_type"), const("agent_cashout")),
              gt(payload("amount"), const(200000)))),
        # FLT_TURN_094: Float turnover > 5x daily limit (potential layering).
        ("FLT_TURN_094_high_turnover", 15,
         gt(fds("float_turnover_ratio"), const(5.0))),
        # FLT_PROV_095: Agent operating in different province than registered.
        ("FLT_PROV_095_province_mismatch", 10,
         eq(fds("agent_province_mismatch"), const(True))),
    ],

    # 3.10 Merchant license compliance — 5 rules, total 100
    "merchant_license_compliance": [
        # MCH_LIC_100: Merchant license expired.
        ("MCH_LIC_100_license_expired", 30,
         neq(payload("merchant_license_status"), const("active"))),
        # MCH_CAT_101: Merchant in high-risk category AND license not SBP-approved.
        ("MCH_CAT_101_high_risk_unapproved", 25,
         and_(eq(fds("merchant_high_risk"), const(True)),
              eq(fds("merchant_sbp_approved"), const(False)))),
        # MCH_VOL_102: Merchant volume spike > 5x 30-day avg.
        ("MCH_VOL_102_volume_spike", 20,
         gt(fds("merchant_volume_ratio_30d"), const(5.0))),
        # MCH_LOC_103: Merchant terminal location != registered.
        ("MCH_LOC_103_terminal_location_mismatch", 15,
         eq(fds("merchant_location_mismatch"), const(True))),
        # MCH_FBR_104: Merchant without FBR registration for high-volume retail.
        ("MCH_FBR_104_no_fbr_reg", 10,
         and_(eq(payload("merchant_category"), const("retail")),
              eq(payload("fbr_ntn_registered"), const(False)))),
    ],

    # 3.11 Login anomaly detection — 4 rules, total 100
    "login_anomaly_detection": [
        # LOGIN_GEO_110: Login from country != Pakistan AND no travel notice.
        ("LOGIN_GEO_110_foreign_no_notice", 30,
         and_(neq(payload("login_country"), const("PK")),
              eq(fds("travel_notice_active"), const(False)))),
        # LOGIN_DEV_111: New device + new IP within 1h (account takeover).
        ("LOGIN_DEV_111_new_device_and_ip", 25,
         and_(eq(fds("login_new_device"), const(True)),
              eq(fds("login_new_ip"), const(True)))),
        # LOGIN_FAIL_112: 5+ failed logins in 10 minutes.
        ("LOGIN_FAIL_112_brute_force", 25,
         gte(fds("failed_logins_10min"), const(5))),
        # LOGIN_VPN_113: Login IP in known_vpn_ips.csv AND high-risk session.
        ("LOGIN_VPN_113_vpn_high_risk_session", 20,
         and_(eq(fds("vpn_usage"), const(True)),
              eq(fds("session_high_risk"), const(True)))),
    ],

    # 3.12 Dormant account reactivation — 3 rules, total 100
    "dormant_account_reactivation": [
        # DORM_AMT_120: First txn after dormancy ≥ PKR 500k.
        ("DORM_AMT_120_high_value_reactivation", 40,
         and_(eq(fds("dormant_reactivation_flag"), const(True)),
              gte(payload("amount"), const(500000)))),
        # DORM_KYC_121: Reactivation without re-KYC (SBP BPD 26/2005).
        ("DORM_KYC_121_no_rekyc", 30,
         and_(eq(fds("dormant_reactivation_flag"), const(True)),
              eq(fds("rekyc_completed"), const(False)))),
        # DORM_PAT_122: Dormant→active→large outflow within 48h.
        ("DORM_PAT_122_outflow_pattern", 30,
         eq(fds("dormant_outflow_pattern"), const(True))),
    ],

    # 3.13 POS transaction compliance — 3 rules, total 100
    "pos_transaction_compliance": [
        # POS_MCC_130: MCC code in restricted list (gambling, crypto, weapons).
        ("POS_MCC_130_restricted_mcc", 40,
         eq(fds("pos_mcc_restricted"), const(True))),
        # POS_AMT_131: POS txn > PKR 200k (high-value retail anomaly).
        ("POS_AMT_131_high_value_pos", 30,
         and_(eq(payload("channel"), const("POS")),
              gt(payload("amount"), const(200000)))),
        # POS_INTL_132: International POS > USD 1,000 equivalent (~PKR 280k)
        # without travel notice (bank-tunable).
        ("POS_INTL_132_intl_no_notice", 30,
         and_(eq(fds("pos_international"), const(True)),
              gte(payload("amount_usd_equiv"), const(1000)),
              eq(fds("travel_notice_active"), const(False)))),
    ],

    # 3.14 Notification failure monitoring — 5 rules, total 100
    "notification_failure_monitoring": [
        # NTF_OTP_140: OTP delivery failed > 3 times in 1 hour.
        ("NTF_OTP_140_otp_failures", 25,
         gt(fds("otp_failures_1h"), const(3))),
        # NTF_SMS_141: SMS notification failure rate > 5% in 1h for one customer.
        ("NTF_SMS_141_sms_high_failure", 20,
         gt(fds("sms_failure_rate_1h"), const(5.0))),
        # NTF_EMAIL_142: Bounced email notifications.
        ("NTF_EMAIL_142_email_bounced", 20,
         eq(fds("email_bounced"), const(True))),
        # NTF_PUSH_143: Failed push during txn confirmation.
        ("NTF_PUSH_143_push_failed_during_tx", 20,
         and_(eq(payload("event_type"), const("txn_confirm")),
              eq(payload("delivery_status"), const("Failed")))),
        # NTF_RISK_144: All 4 channels failed (potential fraud cover).
        ("NTF_RISK_144_all_channels_failed", 15,
         eq(fds("all_notif_channels_failed"), const(True))),
    ],

    # 3.15 Corporate float monitoring — 5 rules, total 100
    "corporate_float_monitoring": [
        # CORP_VOL_150: Corporate volume > 3x 30-day avg.
        ("CORP_VOL_150_volume_3x_avg", 25,
         gt(fds("corp_volume_ratio_30d"), const(3.0))),
        # CORP_BENF_151: Beneficial owner not declared on txn > PKR 5M
        # (SBP AML/CFT Reg 1 + FATF R.24).
        ("CORP_BENF_151_no_beneficial_owner", 25,
         and_(gt(payload("amount"), const(5000000)),
              eq(payload("beneficial_owner_declared"), const(False)))),
        # CORP_KYC_152: Corporate KYC review > 12 months stale.
        ("CORP_KYC_152_stale_kyc", 25,
         eq(fds("corp_kyc_stale_12m"), const(True))),
        # CORP_INVOICE_153: Invoice mismatch on B2B (trade-based ML).
        ("CORP_INVOICE_153_invoice_mismatch", 15,
         eq(fds("corp_invoice_mismatch"), const(True))),
        # CORP_SUBS_154: Transfers to newly-incorporated subsidiaries (<90d).
        ("CORP_SUBS_154_new_subsidiary_transfer", 10,
         eq(fds("counterparty_new_subsidiary"), const(True))),
    ],
}


# ---------------------------------------------------------------------------
# ISLAMIC_RULES — added to RULES_BY_SCENARIO only when ENABLE_ISLAMIC_SCENARIOS=true.
# Pakistan has both conventional and Islamic banks; default is conventional.
# ---------------------------------------------------------------------------
ISLAMIC_RULES: dict[str, list[tuple[str, int, dict]]] = {

    # 3.5 Shariah contract compliance — 4 rules, total 100
    "shariah_contract_compliance": [
        # SHR_TYPE_040: lms_shariah_tag not in shariah_contract_types.csv.
        ("SHR_TYPE_040_invalid_contract", 30,
         and_(is_not_empty(fds("lms_shariah_tag")),
              neq(fds("lms_shariah_tag"), const("MRB")),
              neq(fds("lms_shariah_tag"), const("MDR")),
              neq(fds("lms_shariah_tag"), const("MSH")),
              neq(fds("lms_shariah_tag"), const("IJR")),
              neq(fds("lms_shariah_tag"), const("WKL")),
              neq(fds("lms_shariah_tag"), const("QRD")),
              neq(fds("lms_shariah_tag"), const("SLM")),
              neq(fds("lms_shariah_tag"), const("IST")))),
        # SHR_RIBA_041: Interest > 0 on Islamic-tagged contract (no riba).
        ("SHR_RIBA_041_interest_on_islamic", 30,
         and_(is_not_empty(fds("lms_shariah_tag")),
              gt(payload("interest_rate"), const(0)))),
        # SHR_VIOL_042: Aggregated Shariah violation flag.
        ("SHR_VIOL_042_violation_detected", 25,
         eq(fds("shariah_violation_detected"), const(True))),
        # SHR_TT_043: tt_code mismatch with shariah_model.
        ("SHR_TT_043_tt_code_mismatch", 15,
         eq(fds("shariah_tt_code_mismatch"), const(True))),
    ],

    # 3.16 Wakalah delegation compliance — 4 rules, total 100
    "wakalah_delegation_compliance": [
        # WKL_BLOCK_160: Wakalah revoked → block.
        ("WKL_BLOCK_160_revoked", 30,
         eq(payload("wakalah_revoked"), const(True))),
        # WKL_AMT_161: Wakalah txns > PKR 1M (delegated authority limit).
        ("WKL_AMT_161_over_delegated_limit", 25,
         and_(eq(payload("contract_type"), const("Wakalah")),
              gt(payload("amount"), const(1000000)))),
        # WKL_EXPIRY_162: Wakalah expired.
        ("WKL_EXPIRY_162_expired", 25,
         eq(fds("wakalah_expired"), const(True))),
        # WKL_SCOPE_163: Txn type outside delegated scope.
        ("WKL_SCOPE_163_out_of_scope", 20,
         eq(fds("wakalah_out_of_scope"), const(True))),
    ],
}


if ENABLE_ISLAMIC_SCENARIOS:
    RULES_BY_SCENARIO.update(ISLAMIC_RULES)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def get_jwt_token() -> str:
    """Authenticate via Firebase (real or emulator) -> Marble JWT."""
    # Emulator quirk: serves REST API at /identitytoolkit.googleapis.com/...; real Firebase serves at /v1/...
    if "localhost" in FIREBASE_URL or ":9099" in FIREBASE_URL:
        signin_url = f"{FIREBASE_URL}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    else:
        signin_url = f"{FIREBASE_URL}/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    r = requests.post(
        signin_url,
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "returnSecureToken": True},
        timeout=10,
    )
    r.raise_for_status()
    id_token = r.json()["idToken"]

    r = requests.post(f"{API_URL}/token", headers={"Authorization": f"Bearer {id_token}"}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def api(method: str, path: str, token: str, json_data=None, **kwargs):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return requests.request(method, f"{API_URL}{path}", headers=headers, json=json_data, timeout=30, **kwargs)


# ---------------------------------------------------------------------------
# Rule-pack application
# ---------------------------------------------------------------------------
def fetch_scenarios(token: str) -> dict[str, str]:
    r = api("GET", "/scenarios", token)
    r.raise_for_status()
    scenarios = r.json()
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios") or scenarios.get("data") or []
    return {s["name"]: s["id"] for s in scenarios}


def fetch_custom_lists(token: str) -> dict[str, str]:
    r = api("GET", "/custom-lists", token)
    r.raise_for_status()
    lists = r.json()
    if isinstance(lists, dict):
        lists = lists.get("custom_lists") or lists.get("data") or []
    return {lst["name"]: lst["id"] for lst in lists}


def add_rules_to_scenario(token: str, scenario_name: str, scenario_id: str,
                          rules: list) -> int:
    if not rules:
        return 0

    r = api("POST", "/scenario-iterations", token, {"scenario_id": scenario_id})
    if r.status_code not in (200, 201):
        print("    ERROR creating iteration for " + scenario_name + ": " + str(r.status_code) + " " + r.text[:120])
        return 0
    iteration_id = r.json()["id"]

    api("PATCH", "/scenario-iterations/" + iteration_id, token, {"body": THRESHOLDS})

    added = 0
    for idx, (rule_name, score, formula) in enumerate(rules, 1):
        r = api("POST", "/scenario-iteration-rules", token, {
            "scenario_iteration_id": iteration_id,
            "name": rule_name,
            "description": "Rule " + rule_name + " (+" + str(score) + ")",
            "score_modifier": score,
            "display_order": idx,
            "rule_group": scenario_name,
            "formula_ast_expression": formula,
        })
        if r.status_code in (200, 201):
            added += 1
        else:
            print("    WARN rule " + rule_name + ": " + str(r.status_code) + " " + r.text[:80])

    r = api("POST", "/scenario-iterations/" + iteration_id + "/commit", token)
    if r.status_code not in (200, 201):
        print("    WARN commit: " + str(r.status_code) + " " + r.text[:100])

    r = api("POST", "/scenario-publications", token, {
        "scenario_iteration_id": iteration_id,
        "publication_action": "publish",
    })
    if r.status_code in (200, 201):
        print("  Published " + scenario_name + ": " + str(added) + "/" + str(len(rules)) + " rules live")
    else:
        print("    WARN publish: " + str(r.status_code) + " " + r.text[:100])

    return added


def seed_custom_lists(token: str, list_ids: dict) -> None:
    print("")
    print("=== Seeding custom lists from " + str(SEED_DIR) + " ===")
    if not SEED_DIR.exists():
        print("  WARNING: " + str(SEED_DIR) + " does not exist — skipping seed data load")
        return

    # Skip Islamic seed unless explicitly enabled.
    seed_files = dict(SEED_FILES)
    if not ENABLE_ISLAMIC_SCENARIOS:
        seed_files.pop("shariah_contract_types", None)

    for list_name, csv_file in seed_files.items():
        lid = list_ids.get(list_name)
        if not lid:
            print("  SKIP " + list_name + ": not provisioned on platform")
            continue
        csv_path = SEED_DIR / csv_file
        if not csv_path.exists():
            print("  SKIP " + list_name + ": seed file " + csv_file + " not found")
            continue

        count = 0
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not any(row):
                    continue
                value = "|".join(v.strip() for v in row)
                api("POST", "/custom-lists/" + lid + "/values", token, {"value": value})
                count += 1
        if count:
            print("  Seeded " + list_name + ": " + str(count) + " values")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    total_rules = sum(len(r) for r in RULES_BY_SCENARIO.values())
    print("=" * 60)
    print(" " + PACK_NAME + " — Jurisdiction Rule Pack (Template)")
    print(" " + str(total_rules) + " rules across " + str(len(RULES_BY_SCENARIO)) + " scenarios")
    print(" Islamic scenarios: " + ("ENABLED" if ENABLE_ISLAMIC_SCENARIOS else "disabled (default)"))
    print("=" * 60)
    print("  API:       " + API_URL)
    print("  Firebase:  " + FIREBASE_URL)
    print("  Seed data: " + str(SEED_DIR))

    print("")
    print("--- Authenticating ---")
    token = get_jwt_token()
    print("  JWT obtained (" + token[:30] + "...)")

    print("")
    print("--- Fetching platform scenarios + lists ---")
    scenarios = fetch_scenarios(token)
    print("  Found " + str(len(scenarios)) + " scenarios")
    list_ids = fetch_custom_lists(token)
    print("  Found " + str(len(list_ids)) + " custom lists")

    print("")
    print("--- Applying rules ---")
    total_added = 0
    for scen_name, rules in RULES_BY_SCENARIO.items():
        sid = scenarios.get(scen_name)
        if not sid:
            print("  SKIP " + scen_name + ": scenario not found on platform (run scripts/setup.py first)")
            continue
        added = add_rules_to_scenario(token, scen_name, sid, rules)
        total_added += added

    seed_custom_lists(token, list_ids)

    print("")
    print("=" * 60)
    print(" Pack applied: " + str(total_added) + " rules published")
    print("=" * 60)
    print("")


if __name__ == "__main__":
    main()
