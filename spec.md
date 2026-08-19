# Marble FDS Platform — Specification

**Version:** 2.0 (generic platform spec)
**Status:** Platform + Pakistan reference pack; see `README.md` for current scope

> **This document specifies the generic Marble FDS platform.** Jurisdiction-specific rule details, regulatory mappings, and seed data live in per-jurisdiction packs under `scripts/rules_library/` and `docs/jurisdictions/`. Jurisdiction-specific implementation detail for the Pakistan reference pack is preserved at `docs/jurisdictions/pakistan/`.

---

## Table of Contents

1. [Purpose & Design Principles](#1-purpose--design-principles)
2. [Architecture](#2-architecture)
3. [Data Model (12 Tables)](#3-data-model-12-tables)
4. [Custom Lists (12 Containers)](#4-custom-lists-12-containers)
5. [Scenario Framework (15 Shells)](#5-scenario-framework-15-shells)
6. [Risk Score Model](#6-risk-score-model)
7. [Three-Tier Escalation](#7-three-tier-escalation)
8. [Bridge Service API](#8-bridge-service-api)
9. [Compliance Infrastructure](#9-compliance-infrastructure)
10. [Jurisdiction Packs](#10-jurisdiction-packs)
11. [Deployment & Provisioning](#11-deployment--provisioning)
12. [Implementation Phases](#12-implementation-phases)

---

## 1. Purpose & Design Principles

### 1.1 Purpose

Marble FDS is a **jurisdiction-agnostic Fraud Detection System platform** for financial institutions — banks, EMIs, mobile-money operators, microfinance lenders. It provides a canonical data model, rule-engine integration, ML escalation, compliance plumbing, and auditing, all on top of the open-source [Checkmarble](https://github.com/checkmarble/marble-backend) decision engine.

The platform is **empty by default**: it ships the plumbing but not the rules. Operators deploy it and then either load a shipped **jurisdiction pack** (e.g., the Pakistan reference pack with ~85 SBP/FMU/FATF rules), build their own pack, or configure rules interactively via the Admin UI.

### 1.2 Design Principles

- **Platform / pack separation.** The platform is jurisdiction-agnostic. Rules, thresholds, and reference data are supplied by jurisdiction packs or the operator.
- **Suggestive only.** All FDS outputs are advisory. A Compliance Officer retains final authority via Maker-Checker workflow.
- **Config-driven.** Thresholds, limits, AML typologies, Shariah models — all data in Checkmarble, not hardcoded.
- **Auditable.** Every decision logged to an immutable audit trail. Retention period set per jurisdiction (typically ≥5 years).
- **Multi-tenant ready.** Checkmarble's organization model means each operator gets an isolated rule namespace.
- **Maker-Checker.** Required for all rule edits, threshold changes, and decision overrides.

### 1.3 Risk Band Mapping (canonical 4-tier, platform-wide)

| Score | Band | Checkmarble Outcome | System Action |
|-------|------|---------------------|---------------|
| 0-24 | Safe (Low) | `approve` | Allow transaction, no action |
| 25-39 | Warning (Medium) | `review` | Notify user/admin, compliance review |
| 40-59 | Risk (High) | `block_and_review` | Freeze, alert compliance, escalate to Marbel/GNN |
| 60+ | Critical | `decline` | Auto-block, STR/CTR draft, escalate |

---

## 2. Architecture

### 2.1 High-Level Flow

```
Event Ingestion (REST / ISO-8583 / Mobile App / POS)
       │
       ▼
  Bridge Service (FastAPI, port 8000)
       │
       ├─── /ingest/{table}       → proxy to Checkmarble
       ├─── /decide                → decision + auto-escalation
       └─── /webhooks/checkmarble  → async decision callback
       │
       ▼
  Checkmarble Decision Engine (port 8080)
       │  Evaluates scenario rules, returns score (0-100)
       │
       ▼
  Score < 40? ──YES──> Approve / Review
       │
       NO
       ▼
  Marbel ML Behavioral Engine (port 5000) — XGBoost
       │  Supervised scoring, returns marbel_risk_score
       │
       ▼
  marbel_score < 60? ──YES──> Combined Score → Compliance Queue
       │
       NO
       ▼
  GNN Graph Fraud Engine (port 5001) — HybridGNN
       │  Entity-level analysis, returns enhanced_score
       │
       ▼
  Weighted Combined Score (Checkmarble 40% + Marbel 35% + GNN 25%)
       │
       ▼
  STR/CTR Draft + Audit Trail Entry + Compliance Queue (Maker-Checker)
```

### 2.2 Service Inventory

| Service | Port | Role |
|---------|------|------|
| PostgreSQL | 5432 | Database (all Checkmarble state) |
| Checkmarble API | 8080 | Decision engine backend |
| Checkmarble Worker | — | Async decision processing |
| Checkmarble UI | 3000 | Admin dashboard |
| Bridge | 8000 | Custom FastAPI — escalation, webhooks, proxy |
| Marbel ML | 5000 | XGBoost behavioral scoring |
| GNN ML | 5001 | Graph-based fraud analysis |
| Firebase Auth Emulator | 9099 | Dev authentication |
| Risk Decay Cron | — | Daily score decay |

---

## 3. Data Model (12 Tables)

Canonical field names — jurisdiction-agnostic. Afghan-specific examples (e.g., national ID format, 34-province codes) are in the jurisdiction pack, not the schema.

### 3.1 `individual_users` — 40 fields

Individual customers, typical KYC tiers L1-L2.

Key fields: `user_id`, `full_name`, `national_id_number`, `date_of_birth`, `gender`, `mobile_number`, `email_address`, `address_province`, `address_district`, `address_full`, `face_biometric_registered`, `voice_biometric_registered`, `biometric_retry_count`, `last_login_imei`, `account_status`, `otp_enabled`, `kyc_level`, `kyc_submission_date`, `kyc_status`, `blacklisted_flag`, `last_transaction_amount`, `last_transaction_date`, `ip_address_last_login`, `geo_location_last_login`, `account_login_attempts_24h`, `pin_set`, `face_biometric_verified`, `voice_biometric_verified`, `l2_upgrade_attempts`, `l2_upgrade_status`, `tin_number`, `l2_kyc_documents_submitted`, `l2_kyc_approval_date`, `biometric_update_date`, `multi_device_login_flag`, `risk_score`, `last_otp_sent_date`.

### 3.2 `merchant_users` — 20 fields

Merchant accounts, typical KYC tiers L3-L5. Key fields: `merchant_uid`, `merchant_name`, `merchant_tin`, `business_license_no`, `license_expiry_date`, `kyc_level`, `contact_number`, `email`, `business_category`, `province_code`, `district_code`, `merchant_iban`, `registration_source`, `registration_date`, `status`, `last_activity`, `aml_flag`, `risk_score`, `pos_enabled`, `agent_referral_id`.

### 3.3 `corporate_wallets` — 36 fields

Corporate and NGO entities. Key fields: `corporate_id`, `organization_name`, `org_type` (Gov/NGO/Private), `tin_number`, `business_license_number`, `registration_date`, `license_expiry_date`, `head_office_address`, `province_code`, `district_code`, `authorized_signatory_name`, `signatory_position`, `signatory_nid`, `signatory_contact`, `multi_user_enabled`, `approval_matrix_uploaded`, `bank_account_linked`, `iban_number`, `wallet_id`, `assigned_float_limit`, `used_float_today`, `available_float`, `risk_score`, `compliance_flag`, `staff_kyc_status`, `last_transaction_date`, `gps_location`, `imei_device_id`, `otp_enabled`, `voice_auth_enabled`, `biometric_auth_enabled`, `document_upload_status`, `contract_type`, `shariah_model`, `compliance_verified`, `audit_logs_enabled`.

### 3.4 `agents` — 40 fields

Agent network. Key fields: `agent_id`, `full_name`, `phone_number`, `email`, `gender`, `date_of_birth`, `national_id_number`, `national_id_issue_date`, `national_id_expiry_date`, `province_code`, `district_code`, `village_or_zone`, `agent_role`, `license_type`, `license_number`, `license_expiry_date`, `referring_super_agent`, `imei_number`, `device_model`, `gps_lat`, `gps_long`, `biometric_status`, `voice_biometric`, `face_biometric`, `kyc_level`, `float_limit_afn`, `float_used_afn`, `risk_score`, `status`, `created_at`, `updated_at_field`, `last_login`, `google_auth_enabled`, `wallet_linked`, `iban_number`, `uid_code`, `contract_type`, `shariah_model`, `audit_flag`, `account_status`.

### 3.5 `wakalah_delegations` — 15 fields

Islamic delegation contracts (optional module — enable per deployment). Key fields: `wakalah_id`, `principal_uid`, `wakil_uid`, `delegation_scope`, `permissions_granted`, `contract_type`, `contract_document_url`, `valid_from`, `valid_until`, `revoked`, `revoked_by`, `revoked_at`, `audit_flag`, `last_used_for_txn`, `notes`.

### 3.6 `transactions` — 20 fields (PRIMARY trigger)

Key fields: `transaction_id`, `from_wallet_uid`, `to_wallet_uid`, `amount`, `currency`, `transaction_type`, `transaction_status`, `risk_score`, `timestamp`, `geo_location`, `imei`, `channel`, `initiated_by`, `approved_by`, `approved_at`, `reversal_flag`, `reversal_reason`, `reversal_txn_id`, `notes`, `audit_flag`.

### 3.7 `notification_logs` — 17 fields

Delivery tracking for push/SMS/email/voice. Fields include delivery_status, retry_count, fallback_used, event_type, audit_flag.

### 3.8 `risk_events` — 25 fields

STR/CTR drafts and anomaly records. Fields include risk_category, trigger_type, gps_location, velocity_count, behavior_flag, device_mismatch, geo_mismatch, alert_status, review_status, compliance_action, str_triggered, ctr_triggered.

### 3.9 `audit_trail` — 15 fields

Immutable compliance audit. Fields include action_type, performed_by_uid, performed_by_role, target_uid, target_entity_type, action_description, ip_address, device_imei, geo_location, status_before, status_after, linked_risk_event, audit_verified.

### 3.10 `biometric_logs` — 18 fields

Face/voice enrollment + verification. Fields include biometric_type, enrollment_status, verification_score, device_imei, encryption_hash, retry_count, fallback_used, linked_kyc_level, offline_enrollment.

### 3.11 `float_delegations` — 18 fields

Agent/corporate operational float. Fields include delegated_from_uid, delegated_to_uid, delegation_type, delegated_amount, gps_location, status, risk_limit_exceeded, contract_type, shariah_model.

### 3.12 `fds_input_features` — 40+ fields (pre-computed)

ML-ready features populated by the operator backend before calling `/decide`. Enables time-windowed rules without Checkmarble SUM pivot support.

Categories:
- **Velocity/burst:** `velocity_spike_detected`, `velocity_1h_count`, `withdrawal_1h_sum_afn`, `volume_burst_detected`
- **Geo/device:** `geo_shift_detected`, `geo_shift_distance_km`, `device_trust_score`, `vpn_usage`, `multi_device_login_flag`, `imei_mismatch_flag`
- **Time:** `is_night_tx`, `is_odd_hour_tx`, `tx_hour_utc`
- **Auth abuse:** `pin_reset_count_24h`, `otp_request_count_24h`, `biometric_failure_count`
- **Identifiers/dormancy:** `iban_mismatch_flag`, `merchant_license_expired_flag`, `account_dormancy_flag`
- **Graph/beneficiary:** `linked_wallets_count`, `repeat_beneficiary_flag`, `staff_biometric_mismatch_count`
- **Compliance:** `str_ctr_history_count`, `sanctions_hit`, `blacklisted_flag_current`, `str_history_flag`, `ocr_verified`, `referral_abuse_flag`, `manual_risk_flag`
- **Agent/float:** `float_used_over_daily_avg_pct`, `float_usage_pct`
- **Shariah:** `zakat_qr_violation_flag`, `product_prohibited_tag_flag`, `lms_shariah_tag`, `shariah_violation_detected`
- **TT/APS:** `tt_code`, `role_tier`

### 3.13 Inter-Table Links (12 Foreign Keys)

```
transactions.from_wallet_uid          → individual_users.user_id
transactions.to_wallet_uid            → individual_users.user_id
transactions.to_wallet_uid            → merchant_users.merchant_uid
transactions.initiated_by             → agents.agent_id
transactions.transaction_id           → risk_events.risk_id
transactions.transaction_id           → fds_input_features.transaction_id
agents.agent_id                       → float_delegations.delegated_to_uid
agents.agent_id                       → biometric_logs.wallet_uid
individual_users.user_id              → biometric_logs.wallet_uid
individual_users.user_id              → notification_logs.wallet_uid
wakalah_delegations.principal_uid     → individual_users.user_id
wakalah_delegations.wakil_uid         → agents.agent_id
```

---

## 4. Custom Lists (12 Containers)

Platform creates 12 empty custom lists. Operators seed them via their jurisdiction pack or the Admin UI.

| # | List | Purpose | Seed by |
|---|------|---------|---------|
| 1 | `aml_blacklist` | AML-flagged entities | Jurisdiction pack from FIU sources |
| 2 | `sanctions_watchlist` | International sanctions (UN/OFAC/EU/FATF) | Jurisdiction pack (auto-synced if available) |
| 3 | `internal_blacklist` | Operator-internal compliance blacklist | Operator |
| 4 | `high_risk_provinces` | High-risk geographic regions | Jurisdiction pack |
| 5 | `high_risk_business_categories` | Elevated AML business sectors | Jurisdiction pack |
| 6 | `shariah_contract_types` | Valid Islamic contract models (optional) | Jurisdiction pack if applicable |
| 7 | `valid_province_codes` | Administrative region codes | Jurisdiction pack |
| 8 | `valid_district_codes` | Sub-region codes | Jurisdiction pack |
| 9 | `known_vpn_ips` | VPN/proxy IP ranges | Jurisdiction pack or operator |
| 10 | `blocked_imeis` | Blocked/stolen device IMEIs | Operator |
| 11 | `str_flagged_users` | Previously STR-flagged entities | Operator (auto-populated by bridge) |
| 12 | `agent_float_limits` | Dynamic float caps per agent role/zone | Operator |

---

## 5. Scenario Framework (15 Shells)

The platform provisions 15 scenarios with names, trigger object types, and threshold bands — but **no rules**. Rules are added by the jurisdiction pack.

| # | Scenario | Trigger | Scope |
|---|----------|---------|-------|
| 1 | `kyc_verification_scoring` | `individual_users` | Tier-based KYC, biometric retries |
| 2 | `transaction_risk_scoring` | `transactions` | Velocity, amount, device/geo |
| 3 | `agent_float_monitoring` | `float_delegations` | Float usage, delegation |
| 4 | `agent_gps_imei_compliance` | `agents` | Zone, device lock, license |
| 5 | `aml_compliance_screening` | `transactions` | STR/CTR, sanctions, geographic risk |
| 6 | `biometric_fraud_detection` | `biometric_logs` | Enrollment, liveness, retry |
| 7 | `wakalah_delegation_compliance` | `wakalah_delegations` | Expiry, revocation, consent |
| 8 | `shariah_contract_compliance` | `transactions` | Islamic contract models (opt-in) |
| 9 | `pos_transaction_compliance` | `transactions` | POS IMEI, QR PIN, high-value |
| 10 | `user_onboarding_risk` | `individual_users` | Biometric/PIN/OTP at signup |
| 11 | `merchant_license_compliance` | `merchant_users` | License, TIN, category, AML |
| 12 | `corporate_float_monitoring` | `corporate_wallets` | Limits, compliance, KYC |
| 13 | `notification_failure_monitoring` | `notification_logs` | Delivery, retries, STR alerts |
| 14 | `login_anomaly_detection` | `individual_users` | Attempts, VPN, multi-device |
| 15 | `dormant_account_reactivation` | `individual_users` | Dormancy, large txn after gap |

Each scenario is created with:
- `trigger_object_type`
- Threshold bands: 25 (review), 40 (block_and_review), 60 (decline)
- Empty rule list (populated by jurisdiction pack)

---

## 6. Risk Score Model

### 6.1 Additive scoring

- All rules in a scenario are evaluated against the trigger object
- Each rule whose condition is `true` adds its score modifier to the total
- Final score = SUM(triggered modifiers), clamped to 0-100
- Score maps to outcome per the 4-tier band table above

### 6.2 Per-category weight framework (100-point scale)

The 100-point scale accommodates typical fraud-detection categories. How an operator distributes weight across them is their choice (via rules). A reasonable baseline:

| Category | Typical Max | Example signal |
|----------|------------:|----------------|
| Biometric retry | 10 | `biometric_retry_count >= 3` |
| Velocity check | 10-15 | `velocity_spike_detected` |
| Geo/IP mismatch | 10 | `geo_shift_distance_km > 5` |
| Device mismatch | 15-20 | `imei_mismatch_flag` |
| Night/odd hour | 5-8 | `is_night_tx` or `is_odd_hour_tx` |
| AML blacklist | 15 | `blacklisted_flag_current` |
| Previous STR | 10 | `str_history_flag` |
| OCR/KYC mismatch | 10 | `ocr_verified = false` |
| Float abuse (agent) | 10-18 | `float_used_over_daily_avg_pct > 40` |
| Referral abuse | 5 | `referral_abuse_flag` |
| Manual flag | 5 | `manual_risk_flag` |
| Composite (multi-signal) | 25 | combined AND of above |

### 6.3 Score decay

Daily cron reduces scores for accounts without recent activity:

| Days Clean | Reduction |
|-----------:|-----------|
| 7+ | −5 |
| 14+ | −10 |
| 30+ | Reset to 0 |

Implemented in `bridge/decay/app.py`. Queries Postgres directly for current-version (`valid_until='infinity'`) records, re-ingests updated scores via HTTP.

---

## 7. Three-Tier Escalation

### 7.1 Layer weights

| Layer | Weight | Triggered when |
|-------|-------:|----------------|
| Checkmarble (rules) | 40% | Always (Layer 1) |
| Marbel (ML) | 35% | Checkmarble score ≥ 40 |
| GNN (Graph) | 25% | Marbel score ≥ 60 |

If a layer is unavailable, weight is redistributed proportionally.

### 7.2 Marbel Behavioral Engine

- **Model:** XGBoost v2
- **Training data:** PaySim (~6M mobile-money transactions)
- **Features:** 35 engineered
- **Metrics:** AUC-ROC 0.9994, Precision 100%, Recall 99.8%
- **Status:** Production-ready
- **Artifacts:** `bridge/marbel/artifacts/`

### 7.3 GNN Fraud Engine

- **Architecture:** HybridGNN (GAT + GraphSAGE fusion with deep classifier)
- **Input:** 173 features (166 base + 7 graph augmentation)
- **Current training data:** Elliptic Bitcoin (203K transactions) — **domain mismatch** for mobile money
- **Metrics:** AUC-ROC 0.787, F1 0.278
- **Status:** Functional; stub fallback used in production pending v4 retrain on PaySim or IBM AML synthetic
- **Artifacts:** `bridge/gnn/artifacts/`

### 7.4 Escalation outputs (canonical)

The canonical output payload from escalation — implemented in `bridge/models.py` `EscalationResult`:

```
risk_score, risk_level (Low/Medium/High/Critical),
marbel_triggered, gnn_triggered,
feature_contributions, triggered_rules[],
entity_graph_signals,
final_decision_suggestion (Allow/Monitor/Escalate/SoftHold),
latency_ms, timestamp,
audit_log_reference, escalation_path_id,
aml_risk_flag, str_suggested, ctr_required, sanctions_match,
compliance_violation_type (AML/STR/CTR/Shariah/Velocity/Device/Sanctions),
compliance_alert_reason,
shariah_violation_flag, shariah_anomaly_reason,
assigned_compliance_officer_id
```

---

## 8. Bridge Service API

### 8.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (reports Checkmarble connectivity) |
| `/webhooks/checkmarble` | POST | Async decision callback from Checkmarble |
| `/ingest/{table_name}` | POST | Proxy ingestion to Checkmarble |
| `/decide` | POST | Synchronous decision + auto-escalation |

### 8.2 Idempotency

All write endpoints accept an `Idempotency-Key` header. Repeat requests with the same key return the cached result.

### 8.3 Error model (canonical)

| Code | Meaning |
|------|---------|
| `AGT-401` | Permission denied |
| `AGT-402` | Geo-scope violation |
| `AGT-429` | Velocity limit |
| `FDS-451` | Manual review required |
| `DOC-415` | Invalid document |

Responses use JSON `{code, message, details}`.

---

## 9. Compliance Infrastructure

### 9.1 STR (Suspicious Transaction Report)

Auto-drafted when combined score ≥ 40. Written to `risk_events` with `str_triggered=true`. Must be reviewed within the jurisdiction's STR filing window (e.g., 24h for the jurisdiction's Financial Intelligence Unit (FIU)).

### 9.2 CTR (Currency Transaction Report)

Auto-drafted when amount ≥ jurisdiction threshold (e.g., 500,000 local currency daily aggregate). Written to `risk_events` with `ctr_triggered=true`.

### 9.3 Sanctions screening

`sanctions_watchlist` custom list checked on every decision via pre-computed `sanctions_hit` flag.

### 9.4 Shariah compliance (optional module)

If enabled, the `shariah_contract_compliance` scenario evaluates TT70-77 contract types:

- TT70 Murabaha, TT71 Mudarabah, TT72 Musharakah, TT73 Ijarah
- TT74 Wakalah, TT75 Qard Hasan, TT76 Salam, TT77 Istisna

Violations flagged in `shariah_violation_flag` + `shariah_anomaly_reason`.

### 9.5 Maker-Checker

All rule edits, threshold changes, and decision overrides require dual approval via the Admin UI.

### 9.6 Audit trail

Every decision + compliance action is written to `audit_trail` with retention per-jurisdiction (typically ≥5 years).

---

## 10. Jurisdiction Packs

### 10.1 What a pack contains

A jurisdiction pack is a single Python script under `scripts/rules_library/` that:

1. Connects to the Checkmarble API using the operator's credentials
2. Reads the 15 provisioned scenarios
3. Adds rule definitions (rule ID, score modifier, AST condition)
4. Seeds relevant custom lists with reference data (usually from `seed_data/<pack_name>/*.csv`)
5. Commits + publishes each scenario iteration

### 10.2 Reference pack

**`scripts/rules_library/pakistan.py`** — ~85 rules spanning KYC, AML/STR/CTR, Shariah, agent/float, biometric, wakalah, notification. Regulatory mapping lives at `docs/jurisdictions/pakistan/REGULATORY_MAPPING.md`. This pack is one example of how to operationalize a jurisdiction's regulatory floor on the Marble FDS platform.

### 10.3 Building a new pack

1. Copy the reference pack as a template
2. Replace the 82 rules with the operator's jurisdiction-specific rules
3. Place seed CSVs under `seed_data/<new_pack_name>/`
4. Document the pack with a README + canonical rules doc
5. Publish + run against a freshly-provisioned platform

---

## 11. Deployment & Provisioning

### 11.1 Boot sequence

1. `docker compose up -d` → starts 9 services
2. `./scripts/setup.sh` → provisions the platform (12 tables, 12 links, 12 empty lists, 15 scenario shells, 0 rules)
3. `python scripts/rules_library/<pack>.py` → optionally loads a jurisdiction pack
4. `pytest tests/ -v` → verifies platform integration

### 11.2 Environment

See `.env.example` for the full list. Required: `PG_PASSWORD`, `JWT_SIGNING_KEY`, `SESSION_SECRET`, `CHECKMARBLE_API_KEY`. Rotation procedures in `SECURITY.md`.

### 11.3 ISO-8583 / APS integration (optional)

Supported MTIs: 1100/1110 (auth), 1200 (settlement), 1420 (reversal), 1500/1510 (reconciliation). Private field 48 carries product flags for Islamic finance. Integration points are the `transactions` table (`mti`, `terminal_id`, `rrn`, `stan` fields).

---

## 12. Implementation Status

1. **Platform scaffolding:** 12 tables, 12 links, 12 lists, 15 scenarios
2. **Escalation pipeline:** Marbel + GNN services, bridge logic, risk decay, STR/CTR
3. **Pakistan reference pack:** ~85 rules verified against public SBP/FMU/FATF source documents
4. **Platform generalization:** rule packs are pluggable per-jurisdiction; the platform ships with no hardcoded jurisdiction assumptions
6. **Input validation + error model (pending):** AGT-401/402/429, FDS-451, DOC-415, idempotency_key
7. **Observability (pending):** structlog + asgi-correlation-id, Prometheus metrics
8. **GNN v4 retrain (last):** retrain on PaySim or IBM AML synthetic, then wire real feature extractor
9. **Optional:** pitch/demo materials
