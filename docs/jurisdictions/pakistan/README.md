# Pakistan — Jurisdiction Pack Documentation

This directory contains the source materials and regulatory mapping for the **Pakistan** jurisdiction pack — the platform's reference example of how a jurisdiction pack is built. It is a **public-source-only template**: every rule and threshold traces to a public SBP / FMU / FATF document (see `REGULATORY_MAPPING.md`), no specific bank's internal policy or data was used, and it's meant to be tightened/extended by whichever bank deploys it.

The runnable pack lives at `scripts/rules_library/pakistan.py` — ~85 rules across 15 scenarios (79 rules / 13 scenarios in conventional mode; +8 optional Islamic-banking rules across 2 more scenarios when enabled). Seed data lives at `seed_data/pakistan/`.

## Why this is a template, not a bank-specific pack

This Pakistan pack starts from the regulatory floor (SBP / FMU / AML Act) rather than any single bank's internal risk policy, and is meant to be customized per deploying institution. To turn it into a bank-specific pack:

```bash
cp scripts/rules_library/pakistan.py scripts/rules_library/<yourbank>_pakistan.py
# then tighten thresholds, add bank-specific rules, swap seed CSVs for live feeds
```

## Contents

| File / Folder | What it is |
|---|---|
| `README.md` | This file |
| `REGULATORY_MAPPING.md` | The regulatory floor: every rule mapped to its SBP / FMU / AML Act 2010 / FATF clause, with public source URLs |
| `pdfs/` | (Empty — no bank-specific source PDFs; SBP/FMU regulations are linked from `REGULATORY_MAPPING.md`) |

## Regulatory coverage (public sources only)

- **State Bank of Pakistan (SBP)** — AML/CFT/CPF Regulations 2020 (and 2023 amendments), Branchless Banking Regulations, Customer Due Diligence Framework, Prudential Regulations (Retail / SME / Corporate / Microfinance / Consumer / Agri), Foreign Exchange Manual, Risk Management Guidelines
- **Financial Monitoring Unit (FMU)** — STR/CTR submission formats (goAML XSD), typology reports, AML Act 2010 implementation
- **FATF** — Pakistan Mutual Evaluation Report (2019) + Follow-Up Reports; informed Pakistan's 2018-2022 grey list status; defines EDD scope
- **NADRA** — CNIC format specification, Verisys API requirements
- **PTA (Pakistan Telecommunication Authority)** — IMEI registration, blocked-IMEI lists
- **SBP Shariah Governance Framework** — for the optional Islamic banking scenarios (disabled by default)

## Rule inventory

~85 scoring rules across 15 scenarios, weighted to a 100-point scale per scenario. Of these:
- **~70 mandatory regulatory rules** (directly traceable to a specific SBP / FMU / AML Act clause)
- **~15 bank-tunable rules** (FATF-guidance defaults, bank tightens later)
- **8 optional Islamic-banking rules** (`SHR_*`, `WKL_*` — disabled by default; conventional banks skip)

Full listing in `REGULATORY_MAPPING.md` §3.

Canonical rule ID prefixes (kept consistent across jurisdiction packs so platform code never needs to change):
- `KYC_*` — KYC verification
- `OBR_*` — Onboarding risk
- `CMP_*` — AML compliance (CTR/STR/PEP/sanctions)
- `IBAN_*` / `MOB_*` / `CNIC_*` — Identity format
- `SHR_*` — Shariah compliance (optional)
- `RSK_*` — Risk scoring + behavior
- `BIO_*` — Biometric
- `AGT_*` / `FLT_*` — Agent + float (branchless banking)
- `MCH_*` — Merchant
- `LOGIN_*` — Login anomaly
- `DORM_*` — Dormancy
- `POS_*` — Point of sale
- `NTF_*` — Notifications
- `CORP_*` — Corporate float
- `WKL_*` — Wakalah (optional)

## Relationship to the platform

- Platform (generic, jurisdiction-agnostic) spec: `/spec.md` in repo root
- This pack's regulatory map: `REGULATORY_MAPPING.md` (here)
- Runnable pack script: `/scripts/rules_library/pakistan.py`
- Seed data: `/seed_data/pakistan/`
- Deep scenario tests: `/tests/test_deep_scenarios_pakistan.py`, fixtures at `/tests/fixtures/pakistan/`

## Status

1. ✅ Seed CSVs (12 files) — `seed_data/pakistan/`
2. ✅ Regulatory mapping (`REGULATORY_MAPPING.md`) — every rule traced to its SBP/FMU/AML Act/FATF clause
3. ✅ **Threshold verification** — numerical thresholds verified against the SBP AML/CFT Regulations, FMU guidance, SBP BB Regulations, and SBP FE Manual
4. ✅ **Islamic-banking decision:** the 8 shariah/wakalah rules + 2 scenarios are kept in the pack but **disabled by default**. Conventional banks see no shariah-related rules fire; Islamic banks opt in via `ENABLE_ISLAMIC_SCENARIOS=true`.
5. ✅ Rule script (`scripts/rules_library/pakistan.py`) — ~85 rules implemented and AST-validated
6. ✅ Reference test fixtures (`tests/fixtures/pakistan/`) + deep scenario test suite (`tests/test_deep_scenarios_pakistan.py`)
7. ⏳ End-to-end run against a live provisioned instance (the platform + this pack have not yet been run together end-to-end by a live deployment — see `scripts/setup.sh` in the repo root to provision, then run `python scripts/rules_library/pakistan.py`)

## What the deploying bank still needs to do

1. **Tighten thresholds** based on their internal risk appetite (the regulatory floor is a starting point, not an end state)
2. **Subscribe to a sanctions/PEP feed** — OpenSanctions (free) or World-Check / LexisNexis Bridger / Dow Jones Risk & Compliance (commercial). Replace the seed `aml_blacklist.csv` and `sanctions_watchlist.csv` with the auto-updating feed.
3. **Add institution-specific rules** — e.g., Asaan Account vs full account differential treatment, channel-specific rules
4. **Calibrate Marbel/GNN** on their last 12-24 months of labeled fraud data (PaySim-trained models work as a starting point but bank-specific data improves precision/recall)
5. **Integrate NADRA Verisys** — for live CNIC verification; requires partnership
6. **Integrate 1Link / Raast** — for transaction data feeds; requires partnership

## Notes on the seed CSVs

The 12 CSVs in `seed_data/pakistan/` are deliberately conservative and minimal. Specifically:

- **`aml_blacklist.csv`** — 10 entries, all from publicly-designated UN 1267 / OFAC SDN lists (Hafiz Saeed, Masood Azhar, Lakhvi, etc.). Production feed should pull from OpenSanctions / commercial provider (~thousands of entries).
- **`sanctions_watchlist.csv`** — 12 publicly-designated organizations (LeT, JuD, JeM, TTP, etc.). Production feed should be auto-updating.
- **`high_risk_provinces.csv`** — 4 entries (Balochistan, KP, AJK, GB) flagged from FATF MER. Production should use district-level granularity.
- **`internal_blacklist.csv`** + **`str_flagged_users.csv`** + **`blocked_imeis.csv`** — empty templates. Bank populates from their own systems.
- **`shariah_contract_types.csv`** — 12 AAOIFI contract types. Conventional banks ignore; Islamic banks use.
- **`agent_float_limits.csv`** — defaults for SBP BB Levels 0/1/2 + Retail/District/Super tiers.
