#!/usr/bin/env python3
"""
Marble FDS — Platform Provisioner
=================================
Creates the generic, jurisdiction-agnostic platform scaffolding:
  - 12 data model tables (canonical schema)
  - 12 inter-table links
  - 12 empty custom lists (containers, operator seeds data)
  - 15 scenario shells (name + trigger + threshold bands, NO rules)

After this runs, the platform is live but returns score=0 for every
decision (no rules have fired). To add rules, run a jurisdiction pack:

    python scripts/rules_library/<pack_name>.py

Or build rules interactively via the Admin UI at http://localhost:3000.

Requires:
  - Marble API running at http://localhost:8080
  - Firebase Auth Emulator at http://localhost:9099
  - Python 3.10+ with 'requests' (pip install requests)

Usage:
  python3 scripts/setup.py
"""

import json
import os
import sys
import time
import csv
from pathlib import Path

# AST helpers shared with jurisdiction packs — imported (not redefined here)
# so that when packs build rule conditions, they speak the same dialect.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_helpers import *  # noqa: F401, F403 — re-exported for callers

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("CHECKMARBLE_API_URL", "http://localhost:8080")
FIREBASE_URL = os.environ.get("FIREBASE_URL", "http://localhost:9099")
FIREBASE_PROJECT = os.environ.get("FIREBASE_PROJECT_ID", "test-project")
ADMIN_EMAIL = os.environ.get("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    sys.exit("ERROR: ADMIN_PASSWORD env var is required. Export it or prefix the command, e.g.\n"
             "  ADMIN_PASSWORD='your-secret' python3 scripts/setup.py")
# seed_data/ holds jurisdiction-specific CSVs (populated by rules_library packs).
# The platform provisioner creates empty list containers and does NOT seed them.
SEED_DIR = Path(__file__).resolve().parent.parent / "seed_data"

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def get_jwt_token():
    """Authenticate via Firebase emulator → Marble JWT.

    If login fails we EXIT — do not silently DELETE /accounts and recreate,
    which wipes every Firebase user in the project. To create a new admin,
    set up the Firebase user manually via /accounts:signUp first, or use the
    Marble Admin UI.
    """
    print("  Signing in to Firebase emulator...")
    r = requests.post(
        f"{FIREBASE_URL}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=placeholder",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "returnSecureToken": True},
    )
    if r.status_code != 200:
        sys.exit(
            f"ERROR: Firebase login failed for {ADMIN_EMAIL} (status={r.status_code}).\n"
            f"  Verify ADMIN_PASSWORD matches the user in the emulator.\n"
            f"  To create the admin user, POST to "
            f"{FIREBASE_URL}/identitytoolkit.googleapis.com/v1/accounts:signUp?key=placeholder\n"
            f"  with body {{\"email\":\"{ADMIN_EMAIL}\",\"password\":\"...\",\"returnSecureToken\":true}}"
        )

    id_token = r.json()["idToken"]

    print("  Exchanging for Marble JWT...")
    r = requests.post(f"{API_URL}/token", headers={"Authorization": f"Bearer {id_token}"})
    r.raise_for_status()
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def api(method, path, token, json_data=None, **kwargs):
    """Make authenticated API call."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.request(method, f"{API_URL}{path}", headers=headers, json=json_data, **kwargs)
    return r


def wait_for_api():
    """Wait for Marble API to become ready."""
    print("  Waiting for API...")
    for i in range(60):
        try:
            r = requests.get(f"{API_URL}/liveness", timeout=2)
            if r.status_code == 200:
                print(f"  API ready (attempt {i+1})")
                return
        except requests.ConnectionError:
            pass
        time.sleep(3)
    print("ERROR: API not ready after 180s")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data Model: Tables and Fields
# ---------------------------------------------------------------------------
# Field type mapping: String, Int, Float, Bool, Timestamp
TABLES = {
    "transactions": {
        "description": "Financial transactions (PRIMARY trigger table)",
        "fields": {
            "transaction_id": "String",
            "from_wallet_uid": "String",
            "to_wallet_uid": "String",
            "amount": "Float",
            "currency": "String",
            "transaction_type": "String",
            "transaction_status": "String",
            "risk_score": "Float",
            "timestamp": "Timestamp",
            "geo_location": "String",
            "imei": "String",
            "channel": "String",
            "initiated_by": "String",
            "approved_by": "String",
            "approved_at": "Timestamp",
            "reversal_flag": "Bool",
            "reversal_reason": "String",
            "reversal_txn_id": "String",
            "notes": "String",
            "audit_flag": "Bool",
        },
    },
    "individual_users": {
        "description": "Individual user accounts and KYC data",
        "fields": {
            "user_id": "String",
            "full_name": "String",
            "national_id_number": "String",
            "date_of_birth": "Timestamp",
            "gender": "String",
            "mobile_number": "String",
            "email_address": "String",
            "address_province": "String",
            "address_district": "String",
            "address_full": "String",
            "face_biometric_registered": "Bool",
            "voice_biometric_registered": "Bool",
            "biometric_retry_count": "Int",
            "last_login_imei": "String",
            "device_os": "String",
            "account_status": "String",
            "otp_enabled": "Bool",
            "kyc_level": "String",
            "kyc_submission_date": "Timestamp",
            "kyc_status": "String",
            "biometric_delete_attempts": "Int",
            "blacklisted_flag": "Bool",
            "last_transaction_amount": "Float",
            "last_transaction_date": "Timestamp",
            "transaction_note": "String",
            "ip_address_last_login": "String",
            "geo_location_last_login": "String",
            "account_login_attempts_24h": "Int",
            "pin_set": "Bool",
            "face_biometric_verified": "Bool",
            "voice_biometric_verified": "Bool",
            "l2_upgrade_attempts": "Int",
            "l2_upgrade_status": "String",
            "tin_number": "String",
            "l2_kyc_documents_submitted": "Bool",
            "l2_kyc_approval_date": "Timestamp",
            "biometric_update_date": "Timestamp",
            "multi_device_login_flag": "Bool",
            "risk_score": "Float",
            "last_otp_sent_date": "Timestamp",
        },
    },
    "merchant_users": {
        "description": "Merchant accounts",
        "fields": {
            "merchant_uid": "String",
            "merchant_name": "String",
            "merchant_tin": "String",
            "business_license_no": "String",
            "license_expiry_date": "Timestamp",
            "kyc_level": "String",
            "contact_number": "String",
            "email": "String",
            "business_category": "String",
            "province_code": "String",
            "district_code": "String",
            "merchant_iban": "String",
            "registration_source": "String",
            "registration_date": "Timestamp",
            "status": "String",
            "last_activity": "Timestamp",
            "aml_flag": "Bool",
            "risk_score": "Float",
            "pos_enabled": "Bool",
            "agent_referral_id": "String",
        },
    },
    "corporate_wallets": {
        "description": "Corporate wallet accounts",
        "fields": {
            "corporate_id": "String",
            "organization_name": "String",
            "org_type": "String",
            "tin_number": "String",
            "business_license_number": "String",
            "registration_date": "Timestamp",
            "license_expiry_date": "Timestamp",
            "head_office_address": "String",
            "province_code": "String",
            "district_code": "String",
            "authorized_signatory_name": "String",
            "signatory_position": "String",
            "signatory_nid": "String",
            "signatory_contact": "String",
            "multi_user_enabled": "Bool",
            "approval_matrix_uploaded": "Bool",
            "bank_account_linked": "Bool",
            "iban_number": "String",
            "wallet_id": "String",
            "assigned_float_limit": "Float",
            "used_float_today": "Float",
            "available_float": "Float",
            "risk_score": "Float",
            "compliance_flag": "String",
            "staff_kyc_status": "String",
            "last_transaction_date": "Timestamp",
            "gps_location": "String",
            "imei_device_id": "String",
            "otp_enabled": "Bool",
            "voice_auth_enabled": "Bool",
            "biometric_auth_enabled": "Bool",
            "document_upload_status": "Bool",
            "contract_type": "String",
            "shariah_model": "String",
            "compliance_verified": "Bool",
            "audit_logs_enabled": "Bool",
        },
    },
    "agents": {
        "description": "Agent accounts and compliance data",
        "fields": {
            "agent_id": "String",
            "full_name": "String",
            "phone_number": "String",
            "email": "String",
            "gender": "String",
            "date_of_birth": "Timestamp",
            "national_id_number": "String",
            "national_id_issue_date": "Timestamp",
            "national_id_expiry_date": "Timestamp",
            "province_code": "String",
            "district_code": "String",
            "village_or_zone": "String",
            "agent_role": "String",
            "license_type": "String",
            "license_number": "String",
            "license_expiry_date": "Timestamp",
            "referring_super_agent": "String",
            "imei_number": "String",
            "device_model": "String",
            "gps_lat": "Float",
            "gps_long": "Float",
            "biometric_status": "String",
            "voice_biometric": "Bool",
            "face_biometric": "Bool",
            "kyc_level": "String",
            "float_limit_afn": "Float",
            "float_used_afn": "Float",
            "risk_score": "Float",
            "status": "String",
            "created_at": "Timestamp",
            "updated_at_field": "Timestamp",
            "last_login": "Timestamp",
            "google_auth_enabled": "Bool",
            "wallet_linked": "Bool",
            "iban_number": "String",
            "uid_code": "String",
            "contract_type": "String",
            "shariah_model": "String",
            "audit_flag": "Bool",
            "account_status": "String",
        },
    },
    "wakalah_delegations": {
        "description": "Wakalah delegation contracts",
        "fields": {
            "wakalah_id": "String",
            "principal_uid": "String",
            "wakil_uid": "String",
            "delegation_scope": "String",
            "permissions_granted": "String",
            "contract_type": "String",
            "contract_document_url": "String",
            "valid_from": "Timestamp",
            "valid_until": "Timestamp",
            "revoked": "Bool",
            "revoked_by": "String",
            "revoked_at": "Timestamp",
            "audit_flag": "Bool",
            "last_used_for_txn": "Timestamp",
            "notes": "String",
        },
    },
    "notification_logs": {
        "description": "Notification delivery logs",
        "fields": {
            "notification_id": "String",
            "wallet_uid": "String",
            "user_type": "String",
            "notification_type": "String",
            "delivery_channel": "String",
            "message_title": "String",
            "message_body": "String",
            "sent_timestamp": "Timestamp",
            "delivery_status": "String",
            "retry_count": "Int",
            "read_status": "Bool",
            "read_timestamp": "Timestamp",
            "fallback_used": "Bool",
            "fallback_method": "String",
            "event_triggered_by": "String",
            "event_type": "String",
            "audit_flag": "Bool",
        },
    },
    "risk_events": {
        "description": "Risk scoring events",
        "fields": {
            "risk_id": "String",
            "wallet_uid": "String",
            "agent_id": "String",
            "risk_score": "Float",
            "risk_category": "String",
            "trigger_type": "String",
            "trigger_description": "String",
            "gps_location": "String",
            "imei": "String",
            "velocity_count": "Int",
            "timestamp": "Timestamp",
            "behavior_flag": "Bool",
            "device_mismatch": "Bool",
            "geo_mismatch": "Bool",
            "kyc_retries": "Int",
            "float_usage_pct": "Float",
            "transaction_volume": "Float",
            "alert_status": "String",
            "review_status": "String",
            "compliance_action": "String",
            "str_triggered": "Bool",
            "ctr_triggered": "Bool",
            "notes": "String",
            "created_at": "Timestamp",
            "updated_at_field": "Timestamp",
        },
    },
    "audit_trail": {
        "description": "System audit trail",
        "fields": {
            "audit_id": "String",
            "action_type": "String",
            "performed_by_uid": "String",
            "performed_by_role": "String",
            "target_uid": "String",
            "target_entity_type": "String",
            "action_description": "String",
            "timestamp": "Timestamp",
            "ip_address": "String",
            "device_imei": "String",
            "geo_location": "String",
            "status_before": "String",
            "status_after": "String",
            "linked_risk_event": "String",
            "audit_verified": "Bool",
        },
    },
    "biometric_logs": {
        "description": "Biometric enrollment and verification logs",
        "fields": {
            "biometric_id": "String",
            "wallet_uid": "String",
            "user_type": "String",
            "biometric_type": "String",
            "enrollment_status": "Bool",
            "last_verified": "Timestamp",
            "verification_score": "Float",
            "device_imei": "String",
            "geo_location": "String",
            "encryption_hash": "String",
            "retry_count": "Int",
            "fallback_used": "String",
            "linked_kyc_level": "String",
            "voice_sample_file": "String",
            "face_image_file": "String",
            "offline_enrollment": "Bool",
            "created_at": "Timestamp",
            "updated_at_field": "Timestamp",
        },
    },
    "float_delegations": {
        "description": "Float delegation records",
        "fields": {
            "delegation_id": "String",
            "delegated_from_uid": "String",
            "delegated_to_uid": "String",
            "delegation_type": "String",
            "delegated_amount": "Float",
            "currency": "String",
            "gps_location": "String",
            "imei": "String",
            "status": "String",
            "delegated_by": "String",
            "delegated_at": "Timestamp",
            "expires_at": "Timestamp",
            "float_used_today": "Float",
            "float_balance": "Float",
            "risk_limit_exceeded": "Bool",
            "audit_flag": "Bool",
            "contract_type": "String",
            "shariah_model": "String",
        },
    },
    "fds_input_features": {
        "description": (
            "Pre-computed FDS input features per transaction. "
            "Backend populates these before Checkmarble decision so that "
            "time-windowed and aggregation-based rules can be evaluated "
            "without requiring Checkmarble SUM pivots (not in v0.59). "
            "Canonical source: Data Contract v2 §1 + GNN PDF §1.1."
        ),
        "fields": {
            "transaction_id": "String",
            "transaction_amount": "Float",
            "transaction_type": "String",  # IBFT/POS/QR/APS/WALLET
            "transaction_purpose": "String",
            # Velocity / burst (pre-computed aggregations)
            "biometric_failure_count": "Int",
            "velocity_spike_detected": "Bool",        # RSK_VEL_060: >5 txn/60s
            "velocity_1h_count": "Int",               # CMP_CTR_021: >=10 transfers/1h
            "withdrawal_1h_sum_afn": "Float",         # CMP_STR_020: >100K AFN/1h
            "volume_burst_detected": "Bool",
            # Geo / device
            "geo_shift_detected": "Bool",
            "geo_shift_distance_km": "Float",         # RSK_LOC_061, CMP_GPS_ANOMALY_024
            "device_trust_score": "Float",            # 0-1
            "vpn_usage": "Bool",
            "multi_device_login_flag": "Bool",
            "imei_mismatch_flag": "Bool",             # RSK_DEV_063: current vs last_login_imei
            # Time
            "is_night_tx": "Bool",                    # RSK_TME_062: 23:00-05:00
            "is_odd_hour_tx": "Bool",                 # Risk Scoring.pdf: 00:00-03:00
            "tx_hour_utc": "Int",                     # 0-23 for debugging / Marbel features
            # Auth abuse
            "pin_reset_count_24h": "Int",
            "otp_request_count_24h": "Int",
            # IBAN / license / dormancy
            "iban_mismatch_flag": "Bool",             # IBAN_MOBCHK_042, IBAN_ZONE_041
            "merchant_license_expired_flag": "Bool",
            "account_dormancy_flag": "Bool",
            # Graph / beneficiary / compliance (NEW — Data Contract §1)
            "staff_biometric_mismatch_count": "Int",  # merchant staff KYC issues
            "linked_wallets_count": "Int",            # GNN graph signal
            "repeat_beneficiary_flag": "Bool",        # High repetition to same beneficiary
            "str_ctr_history_count": "Int",           # Historical STR/CTR count for user
            "sanctions_hit": "Bool",                  # Sanctions screen result
            "blacklisted_flag_current": "Bool",       # AML_BLACKLIST: +15
            "str_history_flag": "Bool",               # STR_HISTORY: +10
            "ocr_verified": "Bool",                   # OCR_MISMATCH: +10 if false
            "referral_abuse_flag": "Bool",            # REFERRAL_ABUSE: +5
            "manual_risk_flag": "Bool",               # MANUAL_RISK_FLAG: +5
            # Agent/float (RSK_FLT_065)
            "float_used_over_daily_avg_pct": "Float", # >40% spike → +18
            "float_usage_pct": "Float",               # current float_used / float_limit
            # Shariah (NEW — Data Contract §1)
            "zakat_qr_violation_flag": "Bool",        # Improper QR use for Zakat
            "product_prohibited_tag_flag": "Bool",   # Alcohol/gambling/etc.
            "lms_shariah_tag": "String",              # MUR/MUD/MSH/IJAR/WAK/QH/SALAM/ISTISNA
            "shariah_violation_detected": "Bool",     # Any Shariah rule violated
            # TT + APS (GNN PDF §1.1)
            "tt_code": "Int",                         # 10-77
            "role_tier": "String",                    # L1-L6
        },
    },
}


def create_data_model(token):
    """Create all 12 tables with fields."""
    print("\n=== Step 2: Creating Data Model (12 tables) ===")

    # Track table and field IDs for later use
    table_ids = {}
    field_ids = {}  # {table_name: {field_name: field_id}}

    for table_name, table_def in TABLES.items():
        print(f"  Creating table: {table_name}")
        r = api("POST", "/data-model/tables", token, {
            "name": table_name,
            "description": table_def["description"],
        })
        if r.status_code not in (200, 201):
            print(f"    ERROR: {r.status_code} {r.text}")
            continue

        table_id = r.json()["id"]
        table_ids[table_name] = table_id
        field_ids[table_name] = {}

        # Add fields
        for field_name, field_type in table_def["fields"].items():
            r2 = api("POST", f"/data-model/tables/{table_id}/fields", token, {
                "name": field_name,
                "type": field_type,
                "description": f"{field_name} ({field_type})",
                "nullable": True,
            })
            if r2.status_code in (200, 201):
                fid = r2.json().get("id", "")
                field_ids[table_name][field_name] = fid
            else:
                print(f"    WARN field {field_name}: {r2.status_code} {r2.text[:100]}")

        n_fields = len(table_def["fields"])
        print(f"    OK — {n_fields} fields (+object_id, updated_at)")

    # Fetch full data model to get auto-created field IDs (object_id, updated_at)
    r = api("GET", "/data-model", token)
    if r.status_code == 200:
        dm = r.json().get("data_model", {}).get("tables", {})
        for tname, tdata in dm.items():
            if tname not in field_ids:
                field_ids[tname] = {}
            for fname, fdata in tdata.get("fields", {}).items():
                field_ids[tname][fname] = fdata["id"]
            if tname not in table_ids:
                table_ids[tname] = tdata["id"]

    total_fields = sum(len(f) for f in field_ids.values())
    print(f"\n  Data model complete: {len(table_ids)} tables, {total_fields} fields")
    return table_ids, field_ids


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
LINKS = [
    # Parent = "one" side (object_id has unique constraint), Child = "many" side (FK field)
    # Marble requires parent_field to have active_unique_constraint; only object_id qualifies.
    # Data must use matching values: object_id on parent == FK field value on child.
    ("sender_to_txn", "individual_users", "object_id", "transactions", "from_wallet_uid"),
    ("receiver_to_txn", "individual_users", "object_id", "transactions", "to_wallet_uid"),
    ("merchant_to_txn", "merchant_users", "object_id", "transactions", "to_wallet_uid"),
    ("agent_to_txn", "agents", "object_id", "transactions", "initiated_by"),
    ("txn_to_risk", "transactions", "object_id", "risk_events", "risk_id"),
    ("txn_to_fds", "transactions", "object_id", "fds_input_features", "transaction_id"),
    ("agent_to_float", "agents", "object_id", "float_delegations", "delegated_to_uid"),
    ("agent_to_bio", "agents", "object_id", "biometric_logs", "wallet_uid"),
    ("user_to_bio", "individual_users", "object_id", "biometric_logs", "wallet_uid"),
    ("user_to_notif", "individual_users", "object_id", "notification_logs", "wallet_uid"),
    ("user_to_wakalah", "individual_users", "object_id", "wakalah_delegations", "principal_uid"),
    ("agent_to_wakalah", "agents", "object_id", "wakalah_delegations", "wakil_uid"),
]


def create_links(token, table_ids, field_ids):
    """Create 12 inter-table links."""
    print("\n=== Step 3: Creating Links (12) ===")
    created = 0
    for name, pt, pf, ct, cf in LINKS:
        pt_id = table_ids.get(pt, "")
        ct_id = table_ids.get(ct, "")
        pf_id = field_ids.get(pt, {}).get(pf, "")
        cf_id = field_ids.get(ct, {}).get(cf, "")
        if not all([pt_id, ct_id, pf_id, cf_id]):
            print(f"  SKIP {name}: missing IDs (pt={pt_id}, pf={pf_id}, ct={ct_id}, cf={cf_id})")
            continue

        r = api("POST", "/data-model/links", token, {
            "name": name,
            "parent_table_id": pt_id,
            "parent_field_id": pf_id,
            "child_table_id": ct_id,
            "child_field_id": cf_id,
        })
        if r.status_code in (200, 201, 204):
            created += 1
            print(f"  OK: {name}")
        else:
            print(f"  WARN {name}: {r.status_code} {r.text[:100]}")

    print(f"\n  Links complete: {created}/{len(LINKS)}")


# ---------------------------------------------------------------------------
# Custom Lists
# ---------------------------------------------------------------------------
LIST_DEFS = [
    ("aml_blacklist", "AML-flagged entities — seeded per jurisdiction (FIU/OpenSanctions/etc.)"),
    ("sanctions_watchlist", "International sanctions screening (UN, OFAC, EU, FATF)"),
    ("internal_blacklist", "Operator-internal compliance-flagged entities"),
    ("high_risk_provinces", "Administrative regions classified as high-risk for AML/CFT"),
    ("high_risk_business_categories", "Business categories with elevated AML risk"),
    ("shariah_contract_types", "Valid Islamic contract models (TT70-77) — optional module"),
    ("valid_province_codes", "Valid administrative region codes (primary level)"),
    ("valid_district_codes", "Valid administrative region codes (secondary level)"),
    ("known_vpn_ips", "Known VPN/proxy IP ranges"),
    ("blocked_imeis", "Blocked/stolen/compromised IMEI numbers"),
    ("str_flagged_users", "Users flagged for Suspicious Transaction Reports"),
    ("agent_float_limits", "Dynamic float caps per agent role and zone"),
]


def create_lists(token):
    """Create 12 empty custom list containers.

    The platform creates EMPTY containers only — seeding is the
    jurisdiction pack's responsibility. Packs in scripts/rules_library/
    seed lists from their own CSVs under seed_data/<pack_name>/.
    """
    print("\n=== Step 4: Creating Custom Lists (12 empty containers) ===")
    list_ids = {}
    for name, desc in LIST_DEFS:
        r = api("POST", "/custom-lists", token, {"name": name, "description": desc})
        if r.status_code in (200, 201):
            lid = r.json().get("custom_list", {}).get("id", "")
            list_ids[name] = lid
            print(f"  Created empty list: {name} ({lid[:8]}...)")
        else:
            print(f"  WARN {name}: {r.status_code} {r.text[:100]}")

    print(f"\n  Lists complete: {len(list_ids)} empty containers")
    print("  (Run a jurisdiction pack from scripts/rules_library/ to seed them)")
    return list_ids


# AST Expression Helpers are imported from _ast_helpers.py at the top of
# this file (via ``from _ast_helpers import *``). Jurisdiction packs import
# from the same module so rule conditions speak an identical AST dialect.


# ---------------------------------------------------------------------------
# Scenarios with Rules
# ---------------------------------------------------------------------------
def create_scenarios(token):
    """Create 15 scenarios with iterations, rules, thresholds, and publish."""
    print("\n=== Step 5: Creating Scenarios (15) ===")

    scenarios_created = 0

    # Thresholds: approve 0-24, review 25-39, block_and_review 40-59, decline 60+
    THRESHOLDS = {
        "score_review_threshold": 25,
        "score_block_and_review_threshold": 40,
        "score_decline_threshold": 60,
    }

    # =========================================================================
    # SCENARIO DEFINITIONS (empty shells — jurisdiction packs add rules)
    #
    # The platform creates 15 canonical scenario shells with name, trigger
    # object type, and threshold bands (25 / 40 / 60). Each ships with an
    # EMPTY rule list. To add rules, run a jurisdiction pack from
    # scripts/rules_library/ after this script completes.
    #
    # Scenario names are platform-wide constants — jurisdiction packs look
    # up scenarios by these names to attach their rules.
    # =========================================================================
    scenario_defs = [
        {
            "name": "kyc_verification_scoring",
            "description": "KYC Verification Scoring — tier-based KYC, biometric retries, upgrade failures",
            "trigger": "individual_users",
            "rules": [],
        },
        {
            "name": "transaction_risk_scoring",
            "description": "Transaction Risk Scoring (PRIMARY) — velocity, amount, device/geo, sanctions",
            "trigger": "transactions",
            "rules": [],
        },
        {
            "name": "agent_float_monitoring",
            "description": "Agent Float Monitoring — caps, usage, delegation, Shariah model",
            "trigger": "float_delegations",
            "rules": [],
        },
        {
            "name": "agent_gps_imei_compliance",
            "description": "Agent GPS/IMEI Compliance — zone, device lock, license, biometric, 2FA",
            "trigger": "agents",
            "rules": [],
        },
        {
            "name": "aml_compliance_screening",
            "description": "AML/Compliance Screening — STR/CTR thresholds, sanctions, high-risk regions",
            "trigger": "transactions",
            "rules": [],
        },
        {
            "name": "biometric_fraud_detection",
            "description": "Biometric Fraud Detection — enrollment, liveness, retry, fallback",
            "trigger": "biometric_logs",
            "rules": [],
        },
        {
            "name": "wakalah_delegation_compliance",
            "description": "Wakalah Delegation Compliance — consent, expiry, revocation, contract validity",
            "trigger": "wakalah_delegations",
            "rules": [],
        },
        {
            "name": "shariah_contract_compliance",
            "description": "Shariah Contract Compliance (optional module) — Islamic contract models TT70-77",
            "trigger": "transactions",
            "rules": [],
        },
        {
            "name": "pos_transaction_compliance",
            "description": "POS Transaction Compliance — IMEI lock, QR+PIN, high-value POS",
            "trigger": "transactions",
            "rules": [],
        },
        {
            "name": "user_onboarding_risk",
            "description": "User Onboarding Risk — biometric enrollment, PIN/OTP, VPN login, province",
            "trigger": "individual_users",
            "rules": [],
        },
        {
            "name": "merchant_license_compliance",
            "description": "Merchant License Compliance — TIN, license, AML flag, blacklist, risk score",
            "trigger": "merchant_users",
            "rules": [],
        },
        {
            "name": "corporate_float_monitoring",
            "description": "Corporate Float Monitoring — limits, compliance, staff KYC, approval matrix",
            "trigger": "corporate_wallets",
            "rules": [],
        },
        {
            "name": "notification_failure_monitoring",
            "description": "Notification Failure Monitoring — delivery failures, retries, STR alert failures",
            "trigger": "notification_logs",
            "rules": [],
        },
        {
            "name": "login_anomaly_detection",
            "description": "Login Anomaly Detection — excessive attempts, VPN, multi-device, blacklist",
            "trigger": "individual_users",
            "rules": [],
        },
        {
            "name": "dormant_account_reactivation",
            "description": "Dormant Account Reactivation — dormancy status, expired KYC, outdated biometric",
            "trigger": "individual_users",
            "rules": [],
        },
    ]

    for sdef in scenario_defs:
        name = sdef["name"]
        print(f"\n  Creating scenario: {name}")

        # 1. Create scenario
        r = api("POST", "/scenarios", token, {
            "name": name,
            "description": sdef["description"],
            "trigger_object_type": sdef["trigger"],
        })
        if r.status_code not in (200, 201):
            print(f"    ERROR creating scenario: {r.status_code} {r.text[:100]}")
            continue

        scenario_id = r.json()["id"]

        # 2. Create iteration
        r = api("POST", "/scenario-iterations", token, {"scenario_id": scenario_id})
        if r.status_code not in (200, 201):
            print(f"    ERROR creating iteration: {r.status_code} {r.text[:100]}")
            continue

        iteration_id = r.json()["id"]

        # 3. Update thresholds
        api("PATCH", f"/scenario-iterations/{iteration_id}", token, {"body": THRESHOLDS})

        # 4. Add rules
        for idx, (rule_name, score, formula) in enumerate(sdef["rules"], 1):
            r = api("POST", "/scenario-iteration-rules", token, {
                "scenario_iteration_id": iteration_id,
                "name": rule_name,
                "description": f"Rule {rule_name} (score +{score})",
                "score_modifier": score,
                "display_order": idx,
                "rule_group": name,
                "formula_ast_expression": formula,
            })
            if r.status_code not in (200, 201):
                print(f"    WARN rule {rule_name}: {r.status_code} {r.text[:80]}")

        # 5. Commit iteration (creates a version)
        r = api("POST", f"/scenario-iterations/{iteration_id}/commit", token)
        if r.status_code not in (200, 201):
            print(f"    WARN commit: {r.status_code} {r.text[:100]}")

        # 6. Publish iteration (makes it live for decisions)
        r = api("POST", "/scenario-publications", token, {
            "scenario_iteration_id": iteration_id,
            "publication_action": "publish",
        })
        if r.status_code in (200, 201):
            print(f"    Published: {name} ({len(sdef['rules'])} rules)")
            scenarios_created += 1
        else:
            print(f"    WARN publish: {r.status_code} {r.text[:100]}")
            print(f"    Created (unpublished): {name} ({len(sdef['rules'])} rules)")
            scenarios_created += 1

    print(f"\n  Scenarios complete: {scenarios_created}/{len(scenario_defs)}")


# ---------------------------------------------------------------------------
# Seed Ingestion
# ---------------------------------------------------------------------------
def ingest_bootstrap(token):
    """Ingest one bootstrap record per table to verify the data model."""
    print("\n=== Step 6: Ingesting Bootstrap Records ===")
    records = {
        "transactions": {
            "object_id": "bootstrap-txn-001", "updated_at": "2026-01-30T00:00:00Z",
            "transaction_id": "bootstrap-txn-001", "from_wallet_uid": "bootstrap-ind-001",
            "to_wallet_uid": "bootstrap-mch-001", "amount": 0, "currency": "AFN",
            "transaction_type": "transfer", "transaction_status": "Success",
        },
        "individual_users": {
            "object_id": "bootstrap-ind-001", "updated_at": "2026-01-30T00:00:00Z",
            "user_id": "bootstrap-ind-001", "full_name": "Bootstrap User",
            "kyc_level": "L1", "account_status": "Active", "blacklisted_flag": False,
        },
        "merchant_users": {
            "object_id": "bootstrap-mch-001", "updated_at": "2026-01-30T00:00:00Z",
            "merchant_uid": "bootstrap-mch-001", "merchant_name": "Bootstrap Merchant",
        },
    }

    for table, data in records.items():
        r = api("POST", f"/ingestion/{table}", token, data)
        status = "OK" if r.status_code in (200, 201) else f"WARN ({r.status_code})"
        print(f"  {table}: {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print(" Marble FDS — Platform Provisioner")
    print("=" * 50)
    print(f"  API:           {API_URL}")
    print(f"  Seed-data dir: {SEED_DIR} (packs only)")

    # Step 1: Wait for API
    print("\n=== Step 1: Waiting for API ===")
    wait_for_api()

    # Step 1b: Authenticate
    print("\n=== Step 1b: Authenticating ===")
    token = get_jwt_token()
    print(f"  JWT obtained ({token[:30]}...)")

    # Step 2: Create data model
    table_ids, field_ids = create_data_model(token)

    # Step 3: Create links
    create_links(token, table_ids, field_ids)

    # Step 4: Create empty custom list containers
    list_ids = create_lists(token)

    # Step 5: Create empty scenario shells (no rules)
    create_scenarios(token)

    # Step 6: Bootstrap ingestion (sanity check)
    ingest_bootstrap(token)

    print("\n" + "=" * 50)
    print(" Platform Provisioned — empty and generic")
    print("=" * 50)
    print(f"  Tables:          {len(table_ids)}")
    print(f"  Links:           {len(LINKS)}")
    print(f"  Custom lists:    {len(list_ids)} (empty)")
    print(f"  Scenario shells: 15 (no rules)")
    print()
    print("  Next steps:")
    print("    1. Open Admin UI: http://localhost:3000")
    print("    2. Add rules — pick ONE:")
    print("       (a) Run a jurisdiction pack:")
    print("           python scripts/rules_library/<pack_name>.py")
    print("       (b) Build rules interactively via Admin UI")
    print("    3. Smoke test: POST http://localhost:8000/decide")
    print()


if __name__ == "__main__":
    main()
