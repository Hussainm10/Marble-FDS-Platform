# the operator Pakistan — Canonical Rules & Regulatory Mapping

> **Status:** Template draft, public-source only. No bank consultation has informed this document. Built from publicly available State Bank of Pakistan (SBP), Financial Monitoring Unit (FMU), Federal Board of Revenue (FBR), Pakistan Telecommunication Authority (PTA), and FATF documents. A specific deploying bank should layer their internal risk policy on top — typically tightening thresholds, extending PEP/sanctions feeds, and adding institution-specific rules.

---

## 1. Source Documents (all public)

| # | Document | Issuer | URL pattern | Why it matters |
|---|---|---|---|---|
| S1 | AML/CFT/CPF Regulations 2020 (and 2023 amendments) | SBP | sbp.org.pk → Acts, Ordinances & Regulations → AML/CFT | Mandatory CDD / EDD, STR triggers, sanctions screening duties |
| S2 | Branchless Banking Regulations | SBP | sbp.org.pk → BPRD → Branchless Banking | Per-tier velocity caps (Level 0/1/2), agent rules |
| S3 | Customer Due Diligence (CDD) Framework | SBP | Within S1 | Mandatory KYC fields, risk-based EDD triggers |
| S4 | Prudential Regulations — Retail / SME / Corporate / Microfinance / Consumer / Agri | SBP | sbp.org.pk → BPRD master circulars | Customer-segment risk constraints |
| S5 | Foreign Exchange Manual (chapters on remittance, FCY accounts) | SBP | sbp.org.pk → Foreign Exchange | FCY transaction rules, hundi/hawala typology |
| S6 | Risk Management Guidelines for Pakistani banks | SBP | sbp.org.pk → BPRD | Operational risk framework |
| S7 | Anti-Money Laundering Act 2010 (statutory) | National Assembly / FMU | fmu.gov.pk; pakistancode.gov.pk | Statutory CTR/STR thresholds, predicate offenses |
| S8 | FMU Typology Reports + STR/CTR submission formats (goAML XSD) | FMU | fmu.gov.pk → Reports | Real fraud/ML typologies seen in Pakistan; mandatory STR/CTR fields |
| S9 | FATF Mutual Evaluation Report — Pakistan (2019) + Follow-Up Reports | FATF | fatf-gafi.org | Identifies regulatory gaps (Pakistan was on grey list 2018-2022); informs EDD scope |
| S10 | PTA published lists (blocked IMEIs, registered telecom IDs) | PTA | pta.gov.pk | Telecom-fraud detection inputs |
| S11 | SBP Shariah Governance Framework (if Islamic banking scenarios are enabled) | SBP — Islamic Banking Department | sbp.org.pk → IBD | Validates shariah_contract_types.csv |
| S12 | NADRA published technical specifications (CNIC format, Verisys API) | NADRA | nadra.gov.pk | CNIC validation rules |

---

## 2. Statutory Numerical Thresholds (verified against current SBP/FMU sources)

> **Methodology:** Each value below was verified against the source document at the URL in the citation. Where the original source was a SBP PDF (e.g. Revised AML/CFT Regulations), the threshold appears in the cited Regulation/Para. Date of verification: 2026-04-28.

> **Pakistan FATF status (context):** Pakistan was **removed from the FATF grey list** on 21 October 2022 after fulfilling its 34-item action plan. As of the 2022 follow-up assessment, Pakistan was rated "compliant or largely compliant" on **38 of 40** FATF Recommendations. The regulatory framework summarised here represents that compliant regime.

| Threshold | Verified Value | Source (citation + URL) |
|---|---|---|
| **CTR — Cash Transaction Report** | Single cash txn ≥ **PKR 2,000,000** OR aggregated cash same-day ≥ PKR 2,000,000 | AML Act 2010 §7; SBP AML/CFT Reg 4 §3 ("the CTRs should be reported for the transactions of rupees two million and above"). [SBP AML/CFT](https://www.sbp.org.pk/l_frame/Revised-AML-CFT-Regulations.pdf) |
| **STR — Suspicious Transaction Report** | **No minimum amount**; qualitative — any txn suspected of ML/TF predicate offences | AML Act 2010 §7 |
| **STR filing window** | **"Without delay"** / immediately upon forming suspicion (NOT a fixed N-day window). The half-yearly STR-count *status report* is due to BPRD within 7 days of close of half-year — that is a separate report, not the STR filing itself. | AML Act 2010 §7; SBP AML/CFT Reg 4 §10 |
| **CDD trigger (occasional customer — cash)** | Banks "obtain copy of CNIC while conducting cash transactions above **rupees 0.5 million**" (= PKR 500,000) | SBP AML/CFT Reg 1 §12(a)(i). [SBP AML/CFT](https://www.sbp.org.pk/l_frame/Revised-AML-CFT-Regulations.pdf) |
| **CDD trigger (online walk-in)** | If transaction **exceeds Rs. 100,000** (PKR 100,000) the name and CNIC No. shall be captured in the system and made accessible at beneficiary branch. | SBP AML/CFT Reg 1 §12 |
| **Wire transfer originator info required** | **Regardless of threshold** — Pakistan is *stricter than FATF*. SBP requires originator + beneficial-owner identification on every wire transfer, domestic or cross-border. | SBP AML/CFT Reg 3 ("regardless of threshold"). FATF baseline is USD/EUR 1,000; SBP exceeds it. |
| **Branchless Banking — Level 0** | PKR **25,000/day**, PKR **40,000/month**, PKR **200,000/year** | [SBP Branchless Banking Regulations](https://www.sbp.org.pk/bprd/2019/C10-Branchless-Banking-Regulations.pdf) |
| **Branchless Banking — Level 1 (standard)** | PKR **50,000/day** (exempt for salary credits + payments to trusted merchants e.g. schools, hospitals), PKR **80,000/month** | SBP BB Regulations |
| **Branchless Banking — Level 1 (temporary)** | Daily limit eliminated; per-month raised to **PKR 5,000,000** during the SBP "Go Cashless" Eid-ul-Azha 2025 campaign (19 May – 15 Jun 2025). Reverted afterwards. | [SBP press / Pakistan Observer](https://pakobserver.net/sbp-raises-transaction-limits-launches-go-cashless-campaign-for-eidul-azha-2025/) |
| **Branchless Banking — Level 2** | **Bank-set** — SBP authorises individual banks to set their own Level 2 limits and monitor activity, subject to risk-based controls. Default in `agent_float_limits.csv` is conservative; bank tunes per their policy. | SBP BB Regulations |
| **Cash withdrawal — high-value alert (bank-tunable, not statutory)** | Default ≥ **PKR 500,000** in one hour | Bank-tunable; FATF typology guidance |
| **Foreign Currency cash purchase / sale** | Cash purchase ≥ **USD 500**: biometric verification + purpose declaration + supporting documents required. Sale txn ≥ **USD 1,000**: documentation required. Aggregate ≥ **USD 10,000/day** or **USD 100,000/year**: enhanced documentation. | [SBP FE Manual Ch.6](https://www.sbp.org.pk/fe_manual/chapters/chapter6.htm) + SBP press 8-Nov-2022 |
| **PEP — enhanced due diligence** | **Always EDD** (no threshold). Senior management approval required to establish/continue PEP relationship. No statutory "half of standard" cap; banks set internal monitoring tightening per risk policy. | SBP AML/CFT Reg 1 §29 |
| **Sanctions hit — block** | UN 1267/1988/2231 + OFAC SDN + FATF + SECP/SBP-issued lists → freeze immediately, file STR | AML Act 2010 §9; UNSC Resolutions |
| **Account dormancy** | After **1 year (12 months)** of no customer-initiated operations → classified dormant; only credit transactions allowed; reactivation requires fresh CDD. After **15 years** total inactivity → surrendered to SBP as unclaimed deposit. | SBP BPD Circular 26/2005; Banking Companies Ordinance 1962 §31; Banking Companies (Amendment) Act 2024 |
| **Record retention** | **10 years** from completion of transaction (CTR/STR + supporting docs) | AML Act 2010 §7B; SBP AML/CFT Reg 5 |

> **Note on PKR thresholds:** SBP revises these via circulars (e.g., the May 2025 Eid-ul-Azha temporary BB Level-1 increase). Production deployment should subscribe to SBP BPRD/BPD/PSD circular feeds and update via `agent_float_limits.csv` / rule constants without code changes.

> **Corrections from earlier draft (transparency):** Five thresholds in the previous draft of this document were incorrect or imprecise: (1) wire transfer originator info — drafted as "USD 1,000" but SBP requires "regardless of threshold"; (2) STR filing window — drafted as "7 working days" but is actually "without delay" (the 7-day rule applies to the half-yearly status *count* report only); (3) BB Level 0 monthly — drafted as 50,000 but is 40,000; (4) BB Level 1 — was incomplete; corrected to 50,000/day + 80,000/month standard; (5) FCY cash deposit threshold — drafted as USD 10,000 but is USD 500 cash purchase or USD 1,000 sale txn. Additionally, the PKR 100,000 online walk-in CNIC threshold was missing and has been added.

---

## 3. Planned Rule Inventory (~80 rules, mirroring Afghan pack structure)

15 scenarios, ~80 weighted scoring rules (target 100-point scale per scenario). Rule IDs use prefixes parallel to the Afghan pack so the AST helpers (`scripts/_ast_helpers.py`) and the platform provisioner (`scripts/setup.py`) work unmodified.

### 3.1 KYC verification scoring (5 rules, target weight 100)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `KYC_CNIC_001` | 25 | Block if `cnic_number` doesn't match `^[0-9]{5}-[0-9]{7}-[0-9]$` (NADRA standard format) | S3 + S12 |
| `KYC_NAME_002` | 20 | Block if `customer_name` doesn't match CNIC name (NADRA Verisys mismatch) | S3 + S12 |
| `KYC_AGE_003` | 15 | Block if customer < 18; flag if > 80 (CDD elevated) | S3 |
| `KYC_DOC_004` | 25 | Block if no scanned ID image OR image quality fails OCR | S1 §3.2 |
| `KYC_BIO_005` | 15 | Block if biometric template not registered (NADRA biometric required for Asaan accounts) | S2 |

### 3.2 User onboarding risk (5 rules, weight 100)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `OBR_PEP_010` | 30 | Block if customer name matches PEP list (sanctions_watchlist.csv) | S1 §6 |
| `OBR_SANC_011` | 30 | Block if customer matches `sanctions_watchlist.csv` (UN/OFAC/FATF) | S7 §9 |
| `OBR_GEO_012` | 15 | Flag elevated if home district in `high_risk_provinces.csv` with `risk_level=critical` | S9 (FATF MER) |
| `OBR_OCC_013` | 15 | Flag elevated if occupation in `high_risk_business_categories.csv` with `aml_flag=true` | S1 + S8 |
| `OBR_VEL_014` | 10 | Flag if 3+ accounts opened by same CNIC within 30 days | S8 typology |

### 3.3 AML compliance screening (8 rules, weight 100)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `CMP_CTR_AMT` | 25 | Trigger CTR signal if single cash txn ≥ PKR 2,000,000 | S7 |
| `CMP_CTR_AGG` | 20 | Trigger CTR signal if aggregated cash deposits ≥ PKR 2,000,000 in 24h | S7 |
| `CMP_STR_LRG` | 15 | Trigger STR if cash withdrawal ≥ PKR 500,000 in 1 hour | S1 + bank pref |
| `CMP_STR_STR` | 15 | Trigger STR if 5+ deposits in 24h, each just below CTR threshold (structuring/smurfing) | S8 typology |
| `CMP_PEP_020` | 10 | Block if counterparty matches PEP list | S1 §6 |
| `CMP_SANC_021` | 10 | Block if counterparty matches sanctions list | S7 |
| `CMP_HIGH_BIZ_022` | 5 | Score boost if merchant_category in high_risk_business_categories.csv | S8 |

### 3.4 Identity format validation (5 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `IBAN_PK_030` | 30 | Reject if IBAN doesn't match `^PK[0-9]{2}[A-Z]{4}[0-9]{16}$` | SBP IBAN spec |
| `IBAN_LEN_031` | 10 | Reject if IBAN length != 24 | SBP IBAN spec |
| `MOB_PK_032` | 25 | Reject if mobile doesn't match `^(\+92|0)3[0-9]{9}$` | PTA |
| `CNIC_FMT_033` | 25 | Already covered in KYC_CNIC_001 — reused for txn-time validation | NADRA |
| `EMAIL_034` | 10 | Soft-flag if email format invalid (warning, not block) | best practice |

### 3.5 Shariah contract compliance (4 rules — disabled by default for conventional banks)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `SHR_TYPE_040` | 30 | Block if `lms_shariah_tag` not in `shariah_contract_types.csv` | S11 |
| `SHR_RIBA_041` | 30 | Block if interest_rate > 0 on Islamic-tagged contract | S11 (no riba) |
| `SHR_VIOL_042` | 25 | Block if `shariah_violation_detected = true` | S11 |
| `SHR_TT_043` | 15 | Validate tt_code matches expected for shariah_model | S11 |

### 3.6 Risk scoring & behavior (15 rules, weight 100)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `RSK_VEL_050` | 12 | Flag if velocity_spike_detected = true | S8 typology |
| `RSK_NIGHT_051` | 8 | Flag if is_night_tx = true (between 00:00-05:00 local) AND amount > PKR 100k | bank pref + FATF |
| `RSK_GEO_052` | 10 | Flag if `geo_location` in high-risk province polygon | S9 |
| `RSK_GEO_MISMATCH_053` | 12 | Flag if home_province != txn_province AND no travel notice | S8 |
| `RSK_DEV_054` | 10 | Flag if imei_mismatch_flag = true | S8 (device fraud) |
| `RSK_VPN_055` | 10 | Flag if source IP in `known_vpn_ips.csv` | best practice |
| `RSK_DORM_056` | 8 | Flag if dormant account suddenly active | S8 + dormancy reg |
| `RSK_DEPS_057` | 10 | Flag if 10+ small deposits in 24h (smurfing) | S8 |
| `RSK_ROUND_058` | 5 | Flag if amount is suspiciously round (PKR 99,000 / 199,000 — sub-CTR) | S8 |
| `RSK_TIME_059` | 5 | Flag txns clustered in <60s bursts | S8 |
| `RSK_FCY_060` | 5 | Score boost for FCY transactions (heightened scrutiny) | S5 |
| `RSK_CB_061` | 10 | Flag cross-border transfer to FATF grey/black-listed jurisdiction | S9 |
| `RSK_SHELL_062` | 5 | Flag transfers to known shell-company patterns | S8 |

### 3.7 Biometric fraud detection (6 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `BIO_FAIL_070` | 25 | Block if biometric_failure_count ≥ 3 in 1 hour | S2 |
| `BIO_LIVE_071` | 25 | Block if liveness check failed | best practice |
| `BIO_NEW_072` | 20 | Flag if biometric template re-enrolled within 7 days of previous | S8 |
| `BIO_NADRA_073` | 15 | Block if NADRA Verisys returned mismatch | S12 |
| `BIO_GEO_074` | 10 | Flag if biometric capture location > 100km from home district within 1h of last capture | S8 (impossible-travel) |
| `BIO_CHANNEL_075` | 5 | Flag if biometric used on channel where it shouldn't be | S6 |

### 3.8 Agent + GPS + IMEI compliance (6 rules — for branchless banking)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `AGT_LIC_080` | 25 | Block if agent_license expired or inactive | S2 |
| `AGT_GPS_081` | 20 | Block if agent's GPS coords not in registered service area | S2 |
| `AGT_IMEI_082` | 20 | Block if agent's IMEI changed without re-registration | S2 |
| `FLT_CAP_083` | 15 | Block if agent's float exceeds tier cap from `agent_float_limits.csv` | S2 |
| `AGT_HRS_084` | 10 | Flag txns by agent outside registered operating hours | S2 |
| `AGT_VOL_085` | 10 | Flag agent doing > 50 txns/hour (potential mule) | S8 |

### 3.9 Agent float monitoring (6 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `FLT_DEP_090` | 20 | Flag if agent float top-up > tier cap | S2 |
| `FLT_NIGHT_091` | 15 | Flag agent float movements 00:00-05:00 | S2 |
| `FLT_VEL_092` | 20 | Flag float depletion velocity > 80% in 1 hour | S2 |
| `FLT_CASH_093` | 20 | Flag agent cashout amount > PKR 200k in single txn | S2 |
| `FLT_TURN_094` | 15 | Flag float turnover ratio > 5x daily limit (potential layering) | S8 |
| `FLT_PROV_095` | 10 | Flag if agent operating in different province than registered | S2 |

### 3.10 Merchant license compliance (5 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `MCH_LIC_100` | 30 | Block if merchant_license expired | S6 |
| `MCH_CAT_101` | 25 | Block if merchant_category in `high_risk_business_categories.csv` AND license not from SBP-approved list | S1 |
| `MCH_VOL_102` | 20 | Flag if merchant volume spike > 5x 30-day avg | S8 |
| `MCH_LOC_103` | 15 | Flag if merchant terminal location != registered location | S6 |
| `MCH_FBR_104` | 10 | Flag merchant without FBR registration for high-volume retail | FBR Income Tax Ord. |

### 3.11 Login anomaly detection (4 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `LOGIN_GEO_110` | 30 | Block if login from country != Pakistan AND no travel notice | bank pref |
| `LOGIN_DEV_111` | 25 | Flag if new device + new IP within 1 hour | S8 (account takeover) |
| `LOGIN_FAIL_112` | 25 | Block if 5+ failed logins in 10 minutes | best practice |
| `LOGIN_VPN_113` | 20 | Block if login IP in `known_vpn_ips.csv` AND high-risk session (e.g. fund transfer) | bank pref |

### 3.12 Dormant account reactivation (3 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `DORM_AMT_120` | 40 | Flag if first txn after dormancy ≥ PKR 500k | S8 |
| `DORM_KYC_121` | 30 | Block if reactivation without re-KYC | S1 §3.7 |
| `DORM_PAT_122` | 30 | Flag if dormant→active→large outflow within 48h | S8 typology |

### 3.13 POS transaction compliance (3 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `POS_MCC_130` | 40 | Block if MCC code in restricted list (gambling, crypto exchange, weapons) | S1 |
| `POS_AMT_131` | 30 | Flag POS txn > PKR 200k (high-value retail anomaly) | S8 |
| `POS_INTL_132` | 30 | Flag international POS txn > USD 1,000 equivalent without travel notice | S5 |

### 3.14 Notification failure monitoring (5 rules — operational risk)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `NTF_OTP_140` | 25 | Flag if OTP delivery failed > 3 times in 1 hour | S6 |
| `NTF_SMS_141` | 20 | Flag if SMS notification failure rate > 5% in 1 hour for a customer | S6 |
| `NTF_EMAIL_142` | 20 | Flag bounced email notifications | S6 |
| `NTF_PUSH_143` | 20 | Flag failed push notifications during txn confirmation | S6 |
| `NTF_RISK_144` | 15 | Score-boost if all 4 channels failed (potential fraud cover) | bank pref |

### 3.15 Corporate float monitoring (5 rules)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `CORP_VOL_150` | 25 | Flag if corporate account volume > 3x 30-day avg | S6 |
| `CORP_BENF_151` | 25 | Block if beneficial_owner not declared on transactions > PKR 5M | S1 + S9 (FATF R.24) |
| `CORP_KYC_152` | 25 | Block if corporate KYC review > 12 months stale | S1 |
| `CORP_INVOICE_153` | 15 | Flag invoice-mismatch on B2B transfers (trade-based ML) | S8 |
| `CORP_SUBS_154` | 10 | Flag transfers to newly-incorporated subsidiaries (< 90 days) | S8 |

### 3.16 Wakalah delegation compliance (4 rules — disabled by default for conventional banks)

| Rule ID | Weight | Logic | Source |
|---|---:|---|---|
| `WKL_BLOCK_160` | 30 | Block if `wakalah_revoked = true` | S11 |
| `WKL_AMT_161` | 25 | Flag wakalah txns > PKR 1M (delegated authority limit) | S11 |
| `WKL_EXPIRY_162` | 25 | Block if wakalah expired | S11 |
| `WKL_SCOPE_163` | 20 | Block if txn type outside delegated scope | S11 |

---

## 4. Summary

- **Total rules:** ~85 (target 80, slight over to allow tuning headroom)
- **Mandatory regulatory rules:** ~70 (directly traceable to a specific S1–S12 clause)
- **Bank-tunable rules:** ~15 (sensible defaults from FATF guidance, bank tightens later)
- **Optional Islamic-banking rules:** 8 (`SHR_*` and `WKL_*` — disabled by default; conventional banks skip)
- **Currencies referenced:** PKR (with USD-equivalent thresholds for FCY)
- **Geographic scope:** All 7 administrative units (4 provinces + ICT + AJK + GB)
- **Sanctions sources:** UN 1267/1988/2231, OFAC SDN, FATF, FMU PEP list — all public

This document is the **regulatory floor**. A bank deploying this pack would:
1. Tighten thresholds based on their risk appetite
2. Subscribe to a commercial sanctions/PEP feed (OpenSanctions free-tier, World-Check, LexisNexis Bridger, Dow Jones Risk & Compliance) and replace the seed CSVs with auto-updating feeds
3. Add institution-specific rules (their existing risk policies)
4. Calibrate Marbel/GNN thresholds on their historical labeled fraud data

---

## 5. Status & next steps

- [x] **Thresholds verified** against SBP/FMU/AML Act sources (28 Apr 2026). Five corrections + one new threshold applied; see "Corrections from earlier draft" note in Section 2.
- [x] **Islamic banking decision:** `shariah_contract_compliance` + `wakalah_delegation_compliance` are kept in the pack but **disabled by default**. Conventional banks see no shariah noise. Islamic banks enable them at provisioning time by setting `ENABLE_ISLAMIC_SCENARIOS=true` env var when running the pack script.
- [x] **Pack name:** `pakistan` (kept consistent with the platform's jurisdiction-pack naming convention).
- [x] **Rule script:** `scripts/rules_library/pakistan.py` — 79 rules across 13 scenarios in conventional mode; 87 rules across 15 scenarios with Islamic enabled. All 87 ASTs pass JSON-serializability + structural well-formedness validation.
- [x] **Reference test fixtures:** `tests/fixtures/pakistan/` — 8 JSON files covering canonical platform tables + a 10-scenario regression suite for post-deployment validation.
- [ ] **End-to-end live provisioning** — deferred to deployment time. Requires setting `CREATE_ORG_NAME` and running `scripts/setup.sh` to bootstrap the organization, then running this pack script. Documented in deployment instructions below.

## 6. Deployment procedure

To deploy this pack to a fresh Pakistani-bank Marble instance:

```bash
# 1. Configure .env for the deploying bank
export CREATE_ORG_NAME='YourBank-Pakistan'
export CREATE_ORG_ADMIN_EMAIL='admin@yourbank.example.pk'
export ADMIN_PASSWORD='<strong-password>'

# 2. Bring up the stack (creates the organization + admin user automatically)
docker compose up -d
# Wait for marble-api to be healthy

# 3. Bootstrap admin user in Firebase (real or emulator — see QUICKSTART.md step 4)

# 4. Generate Marble API key from the UI: Settings → API keys → role=API_CLIENT
# Paste into .env as CHECKMARBLE_API_KEY, then: docker compose up -d bridge

# 5. Run the platform provisioner — creates 12 empty tables, 15 scenario shells, 12 list containers
python3 scripts/setup.py

# 6. Run the Pakistan pack — 79 conventional rules + seed CSVs
python3 scripts/rules_library/pakistan.py

# 6b. (Optional, Islamic banks) — adds 8 more rules
ENABLE_ISLAMIC_SCENARIOS=true python3 scripts/rules_library/pakistan.py

# 7. Validate with the regression suite — all 10 decision_test_scenarios should produce expected outcomes
python3 -c "
import json, httpx
suite = json.load(open('tests/fixtures/pakistan/decision_test_scenarios.json'))
# ... (see tests/fixtures/pakistan/README.md for full template)
"
```
