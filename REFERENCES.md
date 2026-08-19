# References — Master Sources & Verifications

This document consolidates **every external source** the Marble FDS platform and its Pakistan reference jurisdiction pack cite for regulatory thresholds, AML typologies, ML datasets, and the underlying decision engine. It is the single place to look up "where does this number / this rule / this assumption come from?"

> **Verification date:** 2026-04-28. Where a numeric threshold appears in a rule, follow the table below to the source URL + clause; if the source has been revised since this date, update the rule constant before deploying.

---

## 1. Platform foundations

| # | Asset | Provider | License | Why we use it |
|---|---|---|---|---|
| P1 | [Checkmarble (`marble-backend` v0.59)](https://github.com/checkmarble/marble-backend) | Checkmarble | [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) | Open-source decision engine. Provides the rules / scenarios / data-model framework that the platform layers on top of. We pin to **v0.59.0** in `docker-compose.yml`. |
| P2 | [Checkmarble frontend](https://github.com/checkmarble/marble-frontend) | Checkmarble | Elastic License 2.0 | Admin UI at `:3000`. Pinned to **v0.59.1**. |
| P3 | Firebase Authentication (real and emulator) | Google | [Firebase Terms of Service](https://firebase.google.com/terms) | JWT issuance for Marble UI + API. Switched from emulator → real Firebase project on 2026-04-28. |
| P4 | PostgreSQL 15/16 | The PostgreSQL Global Development Group | [PostgreSQL License](https://www.postgresql.org/about/licence/) (BSD-style) | Backing store for Checkmarble + bridge. |
| P5 | FastAPI / pydantic / structlog / asgi-correlation-id / prometheus-client | Various OSS authors | MIT / BSD / Apache-2.0 | Bridge service stack. See `bridge/requirements.txt`. |

> **Re-licensing note:** The Elastic License 2.0 permits commercial use, modification, and redistribution under three conditions — no SaaS resale of the engine, no removal of license keys, no alteration of license/copyright notices. Marble FDS's *additions* (the bridge, the rule packs, the seed data, and the docs in this repo) are released under their own license — see `LICENSE` at the project root.

---

## 2. ML training data

| # | Dataset | Source | Used by | Why this dataset |
|---|---|---|---|---|
| M1 | [PaySim (mobile-money simulator)](https://www.kaggle.com/datasets/ealaxi/paysim1) | Edgar A. Lopez-Rojas et al., Kaggle | Marbel XGBoost v2 + HybridGNN v4 | 6.3M synthetic mobile-money transactions matching the *exact* domain (mobile wallet, agent network, P2P/cash-in/cash-out). PaySim is the de-facto academic benchmark for mobile-money fraud detection. **No retraining is required when adding a new jurisdiction** — fraud typology behaviour is universal at the transaction-graph level; jurisdictional differences are captured by the **rules layer** (Checkmarble), not the ML layer. |
| M2 | [PaySim companion paper](https://www.diva-portal.org/smash/get/diva2:955852/FULLTEXT06.pdf) | Lopez-Rojas, Elmir, Axelsson (2016) | Methodology reference | Documents the simulator's parameters (agent network, customer behavior, MerchantPattern, FraudPattern). |

> **Why full retraining isn't required per jurisdiction:** Marbel's 35 features and the GNN's 42 features are jurisdiction-agnostic (amount, velocity, hop counts, account-age, channel mix, etc.). A jurisdiction pack supplies the regulatory layer; PaySim already captures the universal behavioural signal. Re-training only pays off with **real, bank-supplied transaction data** under a data-sharing agreement — see `bridge/training/scripts/README.md` for the retrain procedure if/when such data becomes available.

---

## 3. Pakistan reference pack — sources

The Pakistan pack (`scripts/rules_library/pakistan.py`, 79 conventional + 8 Islamic rules across 13–15 scenarios) is **public-source-only**: no Pakistani bank's internal documents informed it. Every threshold below was verified against the SBP / FMU / NADRA / PTA / FATF source listed. Full clause-by-clause mapping at `docs/jurisdictions/pakistan/REGULATORY_MAPPING.md`.

### 3.1 Source documents

| ID | Document | Issuer | Public URL | Used for |
|---|---|---|---|---|
| **S1** | AML/CFT/CPF Regulations 2020 (with 2023 amendments) | State Bank of Pakistan (SBP) | <https://www.sbp.org.pk/l_frame/Revised-AML-CFT-Regulations.pdf> | Mandatory CDD / EDD, STR triggers, sanctions screening duties |
| **S2** | Branchless Banking Regulations | SBP | <https://www.sbp.org.pk/bprd/2019/C10-Branchless-Banking-Regulations.pdf> | BB Level 0/1/2 velocity caps; agent rules |
| **S3** | Customer Due Diligence Framework | SBP | (Within S1) | KYC fields, risk-based EDD triggers |
| **S4** | Prudential Regulations — Retail / SME / Corporate / Microfinance / Consumer / Agri | SBP | <https://www.sbp.org.pk/publications/prudential/> | Customer-segment risk constraints |
| **S5** | Foreign Exchange Manual (Ch.6 — FCY accounts/remittance) | SBP | <https://www.sbp.org.pk/fe_manual/chapters/chapter6.htm> | FCY transaction rules, hundi/hawala typology |
| **S6** | Risk Management Guidelines for Pakistani banks | SBP | sbp.org.pk → BPRD | Operational risk framework |
| **S7** | Anti-Money Laundering Act 2010 | National Assembly / FMU | <https://fmu.gov.pk> ; <https://pakistancode.gov.pk> | Statutory CTR/STR thresholds, predicate offenses, record-retention duty |
| **S8** | FMU Typology Reports + STR/CTR formats (goAML XSD) | Financial Monitoring Unit (FMU) | <https://fmu.gov.pk> | Pakistan-observed fraud / ML typologies; mandatory STR/CTR fields |
| **S9** | FATF Mutual Evaluation Report — Pakistan (2019) + Follow-Up Reports | Financial Action Task Force | <https://fatf-gafi.org> | Identifies regulatory gaps; informs EDD scope. **Pakistan removed from grey list 21-Oct-2022** (38/40 FATF Recommendations rated compliant or largely compliant). |
| **S10** | Blocked-IMEI / registered-telecom-ID lists | Pakistan Telecommunication Authority (PTA) | <https://pta.gov.pk> | Telecom-fraud detection inputs |
| **S11** | Shariah Governance Framework (for the optional Islamic scenarios) | SBP — Islamic Banking Department | sbp.org.pk → IBD | Validates `shariah_contract_types.csv` |
| **S12** | CNIC format spec + Verisys API | National Database and Registration Authority (NADRA) | <https://nadra.gov.pk> | CNIC validation rules |
| **S13** | Banking Companies (Amendment) Act 2024 | National Assembly | pakistancode.gov.pk | 15-year unclaimed-deposit surrender rule + dormancy reactivation CDD |
| **S14** | SBP press release — Eid-ul-Azha "Go Cashless" 2025 (BB Level-1 temp limit) | SBP | <https://pakobserver.net/sbp-raises-transaction-limits-launches-go-cashless-campaign-for-eidul-azha-2025/> | Reference for the temporary 19-May–15-Jun-2025 BB Level-1 raise to PKR 5M/month |

### 3.2 Verified statutory thresholds (Pakistan)

| Threshold | Value | Source |
|---|---|---|
| CTR — single cash txn or 24h aggregate | **PKR 2,000,000** | AML Act §7; SBP AML/CFT Reg 4 §3 (S1, S7) |
| STR — minimum amount | **None** (qualitative trigger) | AML Act §7 (S7) |
| STR — filing window | **"Without delay"** (the half-yearly STR-count *report* is due 7 days after period close — separate from the STR filing itself) | AML Act §7; SBP AML/CFT Reg 4 §10 (S1) |
| CDD — occasional cash customer | CNIC required for cash txn ≥ **PKR 500,000** | SBP AML/CFT Reg 1 §12(a)(i) (S1) |
| CDD — online walk-in | CNIC capture for txn > **PKR 100,000** | SBP AML/CFT Reg 1 §12 (S1) |
| Wire transfer originator info | **Required regardless of threshold** (Pakistan is *stricter than FATF* baseline of USD 1,000) | SBP AML/CFT Reg 3 (S1) |
| Branchless Banking Level 0 | PKR **25,000/day**, **40,000/month**, **200,000/year** | SBP BB Regs (S2) |
| Branchless Banking Level 1 (standard) | PKR **50,000/day**, **80,000/month** (salary credit + trusted-merchant payments exempt from daily) | SBP BB Regs (S2) |
| Branchless Banking Level 1 (temporary, 19-May–15-Jun-2025) | Daily eliminated; per-month PKR **5,000,000** | SBP press / S14 |
| Branchless Banking Level 2 | **Bank-set, risk-based** | SBP BB Regs (S2) |
| Cash-withdrawal high-value alert (bank-tunable) | Default PKR **500,000 / 1h** | FATF typology (S9) |
| FCY cash purchase (USD) — biometric + purpose declaration | ≥ **USD 500** | SBP FE Manual Ch.6 (S5) |
| FCY sale txn — documentation | ≥ **USD 1,000** | SBP FE Manual Ch.6 (S5) |
| FCY enhanced documentation | Aggregate ≥ **USD 10,000/day** or **USD 100,000/year** | SBP FE Manual + 8-Nov-2022 press (S5) |
| PEP relationship | **Always EDD**; senior management approval | SBP AML/CFT Reg 1 §29 (S1) |
| Sanctions hit | Freeze + STR if name on UN 1267/1988/2231 / OFAC SDN / FATF / SECP/SBP-issued lists | AML Act §9; UNSC Resolutions (S7) |
| Account dormancy | **12 months** no customer-initiated activity → dormant; **15 years** total inactivity → SBP unclaimed deposit | SBP BPD Circ. 26/2005; Banking Cos. Ordinance 1962 §31; Banking Cos. (Amendment) Act 2024 (S13) |
| Record retention | **10 years** from transaction completion | AML Act §7B; SBP AML/CFT Reg 5 (S1, S7) |

### 3.3 Identifier formats (Pakistan)

| Field | Format | Example | Authority |
|---|---|---|---|
| `cnic_number` | `XXXXX-XXXXXXX-X` (13 digits) | `42101-1234567-1` | NADRA (S12) |
| `iban` | 24 chars, `PK` + 2 check + 4 bank + 16 acct | `PK36SCBL0000001123456702` | SBP IBAN spec |
| `mobile_number` | E.164: `+92` + 10 digits | `+923001234567` | PTA (S10) |
| `ntn_number` | `NNNNNNN-N` (FBR National Tax Number) | `1234567-2` | Federal Board of Revenue |
| `secp_registration_no` | `SECP-NPO-YYYY-XXXX` | `SECP-NPO-2018-1234` | Securities & Exchange Commission of Pakistan |

---

## 4. Industry standards and frameworks (cross-cutting)

| ID | Standard | Issuer | Used by |
|---|---|---|---|
| F1 | [FATF 40 Recommendations](https://www.fatf-gafi.org/en/topics/fatf-recommendations.html) | FATF | EDD scope, sanctions integration, STR/CTR baselines |
| F2 | [UNSC Resolutions 1267 / 1988 / 2231 (sanctions lists)](https://www.un.org/securitycouncil/sanctions/information) | UN Security Council | `sanctions_watchlist.csv` seeding |
| F3 | [OFAC Specially Designated Nationals (SDN)](https://ofac.treasury.gov/sdn-list) | US Treasury OFAC | `sanctions_watchlist.csv` seeding |
| F4 | [goAML XSD (FIU reporting format)](https://www.unodc.org/unodc/en/money-laundering/goaml.html) | UNODC | STR/CTR XML output schema used by Pakistan's FMU and other national FIUs |
| F5 | [ISO 8583 (financial messaging)](https://www.iso.org/standard/31628.html) | ISO | Reference for ingestion-channel field shapes |
| F6 | [ISO 20022 — pacs.008 (FI-to-FI Customer Credit Transfer)](https://www.iso20022.org/) | ISO | Reference for wire-transfer originator-info rule (Pakistan SBP Reg 3) |

---

## 5. Open-source dependencies — runtime stack

| Component | Version pin | License | Source |
|---|---|---|---|
| `marble-backend` | v0.59.0 | Elastic License 2.0 | <https://github.com/checkmarble/marble-backend> |
| `marble-frontend` | v0.59.1 | Elastic License 2.0 | <https://github.com/checkmarble/marble-frontend> |
| `firebase-emulator` | latest | Apache-2.0 (Google) | <https://firebase.google.com/docs/emulator-suite> |
| `postgres` | 15 (image) | PostgreSQL | docker.io/postgres:15 |
| `fastapi` | (see `bridge/requirements.txt`) | MIT | <https://github.com/tiangolo/fastapi> |
| `pydantic` | v2.x | MIT | <https://github.com/pydantic/pydantic> |
| `structlog` | latest | Apache-2.0 | <https://github.com/hynek/structlog> |
| `asgi-correlation-id` | latest | MIT | <https://github.com/snok/asgi-correlation-id> |
| `prometheus-client` | latest | Apache-2.0 | <https://github.com/prometheus/client_python> |
| `xgboost` | (see `bridge/training/requirements.txt`) | Apache-2.0 | <https://github.com/dmlc/xgboost> |
| `torch` / `torch-geometric` | (see GNN training reqs) | BSD / MIT | <https://pytorch.org> ; <https://pytorch-geometric.readthedocs.io> |

---

## 6. How to verify a citation

1. **Find the rule** in `scripts/rules_library/<pack>.py` — every rule has a comment with the source citation (e.g. `# CMP_CTR_AMT: AML Act §7; SBP AML/CFT Reg 4 §3`).
2. **Cross-reference** to this `REFERENCES.md` § 3 to find the source-document table.
3. **Open the URL** and search for the cited clause. If the source document has been revised, update the rule constant + bump the verification date in `REGULATORY_MAPPING.md`.
4. **For ML model performance figures**, see `bridge/marbel/artifacts/` and `bridge/gnn/artifacts/` — each model artifact ships next to a `metrics.json` with the held-out-set scores. Training notebooks live at `bridge/training/scripts/`.

---

## 7. Errata + corrections log

> **Pakistan pack — corrections from earlier draft (2026-04-28):**
> 1. Wire-transfer originator threshold — drafted "USD 1,000", corrected to **"regardless of threshold"** (SBP is stricter than FATF baseline).
> 2. STR filing window — drafted "7 working days", corrected to **"without delay"** (the 7-day rule applies to the half-yearly STR-count *status report* only).
> 3. BB Level 0 monthly cap — drafted PKR 50,000, corrected to **PKR 40,000**.
> 4. BB Level 1 — was incomplete, corrected to **PKR 50,000/day + 80,000/month** (standard).
> 5. FCY cash threshold — drafted "USD 10,000 deposit", corrected to **"USD 500 cash purchase / USD 1,000 sale txn"**.
> 6. PKR 100,000 online walk-in CNIC threshold — was missing, **added**.

---

*Last verified: 2026-04-28. If you spot an outdated URL or a revised threshold, please open a PR updating both this file and the corresponding `PDF_RULES_CANONICAL*.md`.*
