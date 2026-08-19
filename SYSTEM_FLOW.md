# Marble FDS — Complete System Flow

**Version:** 1.0
**Status:** Platform + Pakistan reference pack; see `README.md` for current scope

---

## 1. What This Is

**Marble FDS** is a **jurisdiction-agnostic three-tier Fraud Detection System** built on [Checkmarble](https://github.com/checkmarble/marble-backend), an open-source decision engine. It ships as a **platform** — a generic decision engine, data model, ML escalation pipeline, and compliance plumbing — that any banking sector or EMI can deploy. Rules and reference data are **not hardcoded** into the platform; they're supplied by **jurisdiction packs** (see `scripts/rules_library/`).

The platform enforces scoring across categories like KYC, transaction monitoring, agent compliance, AML/sanctions screening, Shariah compliance (opt-in), biometric fraud detection, and more — on a **0-100 risk scoring scale**. Specific rule thresholds, lists, and typologies are configured per-jurisdiction.

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Operator Mobile App / POS / API                      │
│                  (Users, Agents, Merchants, Corporates)               │
└──────────────────┬───────────────────────────────────────────────────┘
                   │ REST / ISO-8583 Events
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         BRIDGE SERVICE (FastAPI)                      │
│                         http://localhost:8000                         │
│                                                                      │
│  Endpoints:                                                          │
│    POST /ingest/{table}     → Forward data to Checkmarble            │
│    POST /decide             → Request decision + auto-escalate       │
│    POST /webhooks/checkmarble → Receive async decision events        │
│    GET  /health             → System health check                    │
└──────────┬──────────────────────────┬────────────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────┐   ┌─────────────────────────────────────┐
│  CHECKMARBLE API    │   │  CHECKMARBLE ADMIN UI               │
│  http://localhost:   │   │  http://localhost:3000              │
│  8080               │   │                                     │
│                     │   │  • Build/edit scenarios & rules     │
│  • Data ingestion   │   │  • Review decisions (Maker-Checker) │
│  • Rule evaluation  │   │  • Manage custom lists              │
│  • Score calculation│   │  • View audit trail                 │
│  • Decision output  │   │  • Configure thresholds             │
└────────┬────────────┘   └─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          PostgreSQL Database                          │
│                          localhost:5432                               │
│                                                                      │
│  Stores: Data model (12 tables), decisions, scenarios, rules,        │
│          custom lists, audit trail, user sessions                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Three-Tier Scoring Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: CHECKMARBLE (Rule-Based)          Weight: 40%             │
│  • 15 scenarios (shells; rules added by jurisdiction pack)          │
│  • Score modifiers (+5 to +25 per rule)                             │
│  • Additive scoring: Final = SUM(all triggered modifiers)           │
│  • Outcome: approve / review / block_and_review / decline           │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2: MARBEL (ML Behavioral Engine)     Weight: 35%             │
│  • Triggered when Checkmarble score >= 40                           │
│  • Supervised XGBoost behavioral scoring                            │
│  • Feature contribution analysis                                    │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3: GNN (Graph Neural Network)        Weight: 25%             │
│  • Triggered when Marbel score >= 60                                │
│  • Entity-level graph analysis                                      │
│  • Shared devices, linked wallets, network patterns                 │
└─────────────────────────────────────────────────────────────────────┘
```

All decisions are **suggestive**. Final authority rests with a Compliance Officer via Maker-Checker workflow.

---

## 3. Services Deployed (Docker Compose)

| # | Service | Container | Port | Purpose |
|---|---------|-----------|------|---------|
| 1 | **PostgreSQL 15** | marble-postgres | 5432 | Persistent database for all Checkmarble state |
| 2 | **Checkmarble API** | marble-api | 8080 | Backend API — ingestion, decisions, rule engine |
| 3 | **Checkmarble Worker** | marble-api-cron | — | Background jobs (async decision processing) |
| 4 | **Checkmarble Frontend** | marble-app | 3000 | Admin UI — scenario builder, decision review |
| 5 | **Firebase Auth Emulator** | firebase-auth | 9099, 4000 | Authentication for dev |
| 6 | **Bridge Service** | bridge | 8000 | Custom FastAPI — escalation, webhooks, proxy |
| 7 | **Marbel ML** | marbel-ml | 5000 | XGBoost behavioral scoring |
| 8 | **GNN ML** | gnn-ml | 5001 | Graph-based fraud analysis |
| 9 | **Risk Decay Cron** | risk-decay-cron | — | Daily risk-score decay |

---

## 4. Platform Data Model (12 Tables, ~280+ fields)

Canonical schema — generic names that work across jurisdictions:

| # | Table | Fields | What It Stores |
|---|-------|--------|----------------|
| 1 | `individual_users` | 40 | Individual customers (L1-L2 KYC) |
| 2 | `merchant_users` | 20 | Merchant accounts (L3-L5 KYC) |
| 3 | `corporate_wallets` | 36 | Corporate/NGO entities — float, compliance |
| 4 | `agents` | 40 | Agent network — GPS, IMEI, float, biometrics |
| 5 | `wakalah_delegations` | 15 | Islamic delegation contracts (optional module) |
| 6 | `transactions` | 20 | **Primary trigger** — all financial transactions |
| 7 | `notification_logs` | 17 | Push/SMS/Email/Voice delivery tracking |
| 8 | `risk_events` | 25 | Detected anomalies and risk signals |
| 9 | `audit_trail` | 15 | Immutable compliance audit log |
| 10 | `biometric_logs` | 18 | Face/voice enrollment and verification |
| 11 | `float_delegations` | 18 | Agent/corporate operational float |
| 12 | `fds_input_features` | 40+ | Pre-computed ML features per transaction |

### Inter-Table Relationships (12 Foreign Key Links)

```
transactions ──── from_wallet_uid ────→ individual_users.user_id
transactions ──── to_wallet_uid ──────→ individual_users.user_id
transactions ──── to_wallet_uid ──────→ merchant_users.merchant_uid
transactions ──── initiated_by ───────→ agents.agent_id
transactions ──── transaction_id ─────→ risk_events.risk_id
transactions ──── transaction_id ─────→ fds_input_features
agents ────────── agent_id ───────────→ float_delegations
agents ────────── agent_id ───────────→ biometric_logs
individual_users ─ user_id ───────────→ biometric_logs
individual_users ─ user_id ───────────→ notification_logs
wakalah_delegations ─ principal_uid ──→ individual_users
wakalah_delegations ─ wakil_uid ──────→ agents
```

---

## 5. Custom Lists (12 Reference Lists — empty containers)

The platform provisions 12 empty custom lists. Operators seed them via their jurisdiction pack or the Admin UI:

| # | List Name | Purpose |
|---|-----------|---------|
| 1 | `aml_blacklist` | AML-flagged entities (per operator FIU sources) |
| 2 | `sanctions_watchlist` | International sanctions (UN, OFAC, EU, FATF) |
| 3 | `internal_blacklist` | Operator-internal compliance-flagged users |
| 4 | `high_risk_provinces` | High-risk geographic regions |
| 5 | `high_risk_business_categories` | Elevated fraud/AML business sectors |
| 6 | `shariah_contract_types` | Valid Islamic contract models (TT70-77, opt-in) |
| 7 | `valid_province_codes` | Administrative region codes per local regulator |
| 8 | `valid_district_codes` | Sub-region codes for geo-binding |
| 9 | `known_vpn_ips` | VPN/proxy IP ranges |
| 10 | `blocked_imeis` | Blocked/stolen device IMEIs |
| 11 | `str_flagged_users` | Previously STR-flagged entities |
| 12 | `agent_float_limits` | Dynamic float caps per agent role/zone |

---

## 6. The 15 Scenario Shells

The platform creates 15 scenarios with names, trigger object types, and threshold bands — **but no rules**. Rules are added by the jurisdiction pack or via the Admin UI.

| # | Scenario | Trigger Table | Typical Concerns |
|---|----------|---------------|------------------|
| 1 | KYC Verification Scoring | `individual_users` | Tier-based KYC, biometric retries |
| 2 | **Transaction Risk Scoring** | `transactions` | Velocity, amount, device/geo, blacklist |
| 3 | Agent Float Monitoring | `float_delegations` | Float usage, spike, Shariah |
| 4 | Agent GPS/IMEI Compliance | `agents` | Zone, device lock, license, 2FA |
| 5 | AML/Compliance Screening | `transactions` | STR/CTR, sanctions, high-risk regions |
| 6 | Biometric Fraud Detection | `biometric_logs` | Enrollment, liveness, retry |
| 7 | Wakalah/Delegation Compliance | `wakalah_delegations` | Expiry, revocation, contract |
| 8 | Shariah Contract Compliance (opt-in) | `transactions` | Islamic contract models |
| 9 | POS Transaction Compliance | `transactions` | POS IMEI, QR PIN, high-value POS |
| 10 | User Onboarding Risk | `individual_users` | Biometric, PIN, OTP, province |
| 11 | Merchant License Compliance | `merchant_users` | License, TIN, category, AML |
| 12 | Corporate Float Monitoring | `corporate_wallets` | Limits, compliance, KYC |
| 13 | Notification Failure Monitoring | `notification_logs` | Delivery, retries, STR alerts |
| 14 | Login Anomaly Detection | `individual_users` | Attempts, device, VPN |
| 15 | Dormant Account Reactivation | `individual_users` | Dormancy, large txn after gap |

---

## 7. How Checkmarble Works — The Decision Engine

### 7.1 Rule Evaluation

Checkmarble uses a **score-based decision model**. Each scenario contains rules, and each rule has a **score modifier** (e.g., +5, +10, +20). When a trigger event occurs:

1. All rules in the scenario are evaluated against the trigger object
2. Every rule whose condition is `true` adds its score modifier to the total
3. The **final score** = SUM of all triggered rule modifiers
4. The score maps to an **outcome**:

```
Final Score = SUM(all triggered rule modifiers)

IF Final Score >= 60  →  outcome = "decline"         (Critical / Red)
IF Final Score >= 40  →  outcome = "block_and_review" (High Risk / Orange)
IF Final Score >= 25  →  outcome = "review"          (Warning / Yellow)
IF Final Score < 25   →  outcome = "approve"         (Safe / Green)
```

### 7.2 Risk Band Mapping (canonical 4-tier)

| Score | Band | Checkmarble Outcome | System Action |
|-------|------|---------------------|---------------|
| 0-24 | Safe (Low) | `approve` | Allow, no action |
| 25-39 | Warning (Medium) | `review` | Notify user/admin, compliance review |
| 40-59 | Risk (High) | `block_and_review` | Freeze, alert compliance, escalate |
| 60+ | Critical | `decline` | Auto-block, STR/CTR draft, escalate |

### 7.3 Score Accumulation Example

A transaction arrives with these characteristics:
- Velocity spike (>5 txns / 60s): **+15**
- IMEI mismatch from last login: **+20**
- Night-time transaction (23:00-05:00): **+8**
- Large withdrawal (>100K local currency in 1 hour): **+15**

**Total Score = 58** → Outcome: `block_and_review` (High Risk / Orange) → escalates to Marbel + GNN.

---

## 8. Transaction Flow — Step by Step

### 8.1 Normal Transaction (Score < 40)

```
Step 1: USER INITIATES TRANSACTION
   │  User sends money via App/POS/QR/USSD
   ▼
Step 2: DATA INGESTION
   │  Backend → Bridge Service
   │  POST /ingest/transactions {full transaction payload}
   ▼
Step 3: BRIDGE FORWARDS TO CHECKMARBLE
   │  Bridge proxies to Checkmarble API
   │  Checkmarble stores record in PostgreSQL
   ▼
Step 4: DECISION REQUEST
   │  Bridge → POST /decide
   │  Includes scenario_id + trigger_object
   ▼
Step 5: CHECKMARBLE EVALUATES RULES
   │  All rules in "transaction_risk_scoring" run
   │  Each rule independently TRUE/FALSE
   │  Score = SUM(triggered modifiers)
   ▼
Step 6: RESPONSE TO BACKEND
   │  {decision: {outcome: "approve", score: 0, rules_triggered: []}}
   ▼
Step 7: TRANSACTION COMPLETES
   Transaction processed normally. No alerts.
```

### 8.2 Risky Transaction (Score >= 40 — Escalation)

```
Step 1: SUSPICIOUS TRANSACTION ARRIVES
   ▼
Step 2: DATA INGESTION — enriched with fds_input_features
   │  velocity_spike_detected=true, imei_mismatch_flag=true,
   │  is_night_tx=true, withdrawal_1h_sum_afn=120000
   ▼
Step 3: CHECKMARBLE EVALUATES RULES
   │  ✓ RSK_VEL_060 velocity_spike: +15
   │  ✓ RSK_DEV_063 imei_mismatch: +20
   │  ✓ RSK_TME_062 night_transaction: +8
   │  ✓ CMP_STR_020 withdrawal>100K/hr: +15
   │
   │  Score = 58 → Outcome = "block_and_review"
   ▼
Step 4: BRIDGE DETECTS SCORE >= 40 → ESCALATION
   ▼
Step 5: LAYER 2 — MARBEL ML ENGINE
   │  Bridge → POST http://marbel:5000/evaluate
   │  Marbel analyzes behavioral patterns via XGBoost
   │  Returns: marbel_risk_score = 65
   ▼
Step 6: MARBEL SCORE >= 60 → GNN ESCALATION
   ▼
Step 7: LAYER 3 — GNN GRAPH ENGINE
   │  Bridge → POST http://gnn:5001/evaluate
   │  GNN analyzes: shared devices, linked wallets, self-circulation
   │  Returns: enhanced_score = 72
   ▼
Step 8: COMBINED SCORE CALCULATION
   │
   │  Weighted (proportional redistribution if a layer unavailable):
   │  • Checkmarble (40%): 58 × 0.40 = 23.2
   │  • Marbel (35%):      65 × 0.35 = 22.75
   │  • GNN (25%):         72 × 0.25 = 18.0
   │
   │  Combined Score = 64.0
   ▼
Step 9: FINAL DECISION
   │  Combined = 64 → Risk Level: "Critical"
   │  Suggestion: "Escalate" (or "SoftHold")
   │  AML Risk Flag: TRUE
   │  STR Suggested: TRUE
   ▼
Step 10: COMPLIANCE QUEUE
   │  Transaction FROZEN
   │  STR (Suspicious Transaction Report) draft generated
   │  Alert visible in Admin UI → Compliance dashboard
   ▼
Step 11: MAKER-CHECKER REVIEW
   │  Compliance Officer reviews in Admin UI
   │  Final decision: approve / reject / escalate / Shariah review
   │  All decisions are SUGGESTIVE — human retains final authority
   ▼
Step 12: AUDIT TRAIL
   Decision logged in audit_trail (retention period per-jurisdiction, typically ≥5y)
```

### 8.3 Webhook Flow (Async Decision Notification)

```
Step 1: Checkmarble makes a decision (async via worker)
   ▼
Step 2: Checkmarble → POST bridge/webhooks/checkmarble
   │  {event: "decision.created", decision_id, outcome, score}
   ▼
Step 3: Bridge checks score >= 40?
   │  YES → Fetch full decision → Trigger escalation (Marbel → GNN)
   │  NO  → Acknowledge and log
   ▼
Step 4: Bridge returns acknowledgement
```

---

## 9. Automated Provisioning

`./scripts/setup.sh` runs:

```
Step 1: 01_wait_for_api.sh
   └─ Health check loop until Checkmarble API responds

Step 2: scripts/setup.py create_data_model()
   └─ POST 12 tables + fields to /data-model
   └─ All 280+ fields registered with correct types

Step 3: scripts/setup.py create_links()
   └─ Creates 12 inter-table foreign key relationships

Step 4: scripts/setup.py create_lists()
   └─ Creates 12 empty custom lists (containers only — no seeding)

Step 5: scripts/setup.py create_scenarios()
   └─ Creates 15 scenario shells with triggers + thresholds
   └─ NO rules (empty rule list per scenario)

Step 6: scripts/setup.py ingest_bootstrap()
   └─ Ingests one record per table to verify schema
```

Then operators run either:
- A shipped rule pack: `python scripts/rules_library/pakistan.py`
- Their own pack: `python scripts/rules_library/<their_pack>.py`
- Or build rules interactively via Admin UI

---

## 10. Platform-Level Compliance Hooks

| Hook | Behavior | Configuration |
|------|----------|---------------|
| **STR filing** | Auto-drafts on combined score ≥ threshold (typically 40) | Threshold per-jurisdiction |
| **CTR filing** | Amount-based (single txn or daily aggregate) | Threshold configurable (e.g., 500,000 local currency) |
| **Shariah compliance** | Opt-in module — TT70-77 contract validation | Enabled per deployment |
| **Audit retention** | Immutable audit log | Retention period per-jurisdiction |
| **Sanctions screening** | `sanctions_watchlist` lookup on every decision | Populated per-jurisdiction |
| **Maker-Checker** | All escalations route to Compliance Officer queue | Required for all FDS outputs |
| **Risk score decay** | Daily cron reduces scores for clean accounts | 7d: -5, 14d: -10, 30d: reset |

---

## 11. Risk Score Weight Framework (100-Point Scale)

The scale is 0-100. How weights distribute is **operator-determined** via rules. Typical categorization:

| Category | Typical Max | Notes |
|----------|------------:|-------|
| Biometric retry | 10 | 1 retry: +2, ≥3: +10 |
| Velocity check | 10-15 | >5 txns/hour or per-minute |
| Geo/IP mismatch | 10 | Shift, VPN, mismatch |
| Float abuse (agent) | 10-18 | >90% or >40% above avg |
| AML blacklist | 15 | List match |
| Odd hour | 5-8 | 00:00-03:00 or 23:00-05:00 |
| Previous STR | 10 | Historical flag |
| OCR mismatch | 10 | KYC document mismatch |
| Device mismatch | 15-20 | IMEI change |
| Referral abuse | 5 | Spam referrals |
| Manual flag | 5 | Compliance officer override |
| Composite | 25 | Multiple high-risk conditions |

---

## 12. Risk Score Decay (Cron-Driven)

Checkmarble doesn't natively support time-based decay, so the platform runs a daily cron:

| Days Without Incident | Score Reduction |
|----------------------|----------------|
| 7+ days clean | -5 points |
| 14+ days clean | -10 points |
| 30+ days clean | Reset to 0 |

Decay re-ingests updated records (Checkmarble creates a new temporal version via `valid_until='infinity'`).

---

## 13. Localhost Access Points

| Service | URL |
|---------|-----|
| **Checkmarble Admin UI** | http://localhost:3000 |
| **Checkmarble API** | http://localhost:8080 |
| **Bridge Service** | http://localhost:8000 |
| **Firebase Auth Emulator** | http://localhost:9099 |
| **Firebase Emulator UI** | http://localhost:4000 |
| **Marbel ML** | http://localhost:5000 |
| **GNN ML** | http://localhost:5001 |
| **PostgreSQL** | localhost:5432 |

Bootstrap admin credentials are set via `.env` (`CREATE_ORG_ADMIN_EMAIL`).

---

## 14. Summary — What the Platform Provides

1. **Canonical 12-table data model** covering individuals, merchants, corporates, agents, transactions, risk events, audit trail, biometrics, delegations
2. **12 empty custom lists** as containers (operator seeds)
3. **15 scenario shells** with triggers + thresholds (operator adds rules)
4. **12 inter-table relationships** for cross-entity rule evaluation
5. **Three-tier scoring pipeline** (Rules → ML → Graph) with weighted combination
6. **Platform-level compliance infrastructure** — STR/CTR auto-drafting, audit trail, score decay
7. **Maker-Checker workflow** — all decisions are suggestive
8. **Pluggable rule packs** via `scripts/rules_library/` — ship or build your own
9. **Pluggable seed data** via `seed_data/<pack>/` — jurisdiction-specific CSVs
10. **ML model artifacts** trained on public fraud datasets (Marbel: PaySim; GNN: pending v4 retrain)
11. **Automated provisioning** — `./scripts/setup.sh` bootstraps the entire platform
