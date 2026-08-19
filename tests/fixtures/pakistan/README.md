# the operator Pakistan — Reference Test Payloads

These JSON files are reference test payloads for the **the operator Pakistan** jurisdiction pack (`scripts/rules_library/pakistan.py`). They demonstrate realistic ingestion payloads for each of the 12 platform tables, flavored with Pakistani values (Lahore/Karachi/Islamabad addresses, PKR currency, +92 phone numbers, NADRA CNIC format, NTN/SECP IDs, etc.) plus a `decision_test_scenarios.json` containing 10 canonical risk scenarios for post-deployment validation.

These fixtures are **not loaded by platform tests** — they're here as:

- Reference for developers writing their own jurisdiction packs (copy + adapt values)
- Example input shape for the canonical platform data tables
- Sanity-check payloads for manual ingestion + `/decide` testing
- Regression scenarios to validate after rule-pack deployment

## Files

| File | Table / Purpose | Notes |
|------|---|---|
| `individual_users.json` | `individual_users` | Pakistani user with NADRA CNIC + Pakistan IBAN |
| `merchant_users.json` | `merchant_users` | Lahore retail merchant with NTN + business license |
| `corporate_wallets.json` | `corporate_wallets` | Karachi NPO with SECP registration + beneficial owner declared |
| `agents.json` | `agents` | Punjab BB-Level-2 agent |
| `transactions.json` | `transactions` | Sample low-risk PKR transfer |
| `risk_events.json` | `risk_events` | STR draft for FMU goAML filing |
| `fds_input_features.json` | `fds_input_features` | Pre-computed ML features (~70 fields used by Pakistan rules) |
| `decision_test_scenarios.json` | n/a | **10 named test scenarios** for post-deployment regression testing |

## `decision_test_scenarios.json` — the regression suite

10 named scenarios covering the major rule activation paths in the pack. Each entry has:
- `name` — short identifier
- `description` — what the scenario tests
- `expected_outcome` — what rules should fire, whether to escalate, regulatory citation
- `request` — full `/decide` request body ready to POST

Scenarios:

| # | Scenario | Tests rule(s) | Should escalate? |
|---|---|---|---|
| 1 | `safe_low_value_pkr_transfer` | (no rules fire — baseline approval) | no |
| 2 | `ctr_threshold_breach_2M_cash` | `CMP_CTR_AMT_single_2M` | yes (CTR) |
| 3 | `high_velocity_withdrawal_in_1h` | `CMP_STR_LRG_withdrawal_500k_1h` | yes |
| 4 | `pep_match_block` | `CMP_PEP_020_counterparty_pep` | yes |
| 5 | `sanctions_hit_critical` | `CMP_SANC_021_counterparty_sanctions` | yes |
| 6 | `high_risk_geo_balochistan_border` | `RSK_GEO_052_high_risk_geo` | yes |
| 7 | `night_high_value_transfer` | `RSK_NIGHT_051_night_high_value` | no (low score) |
| 8 | `structuring_smurfing_pattern` | `CMP_STR_STR_structuring` + `RSK_ROUND_058` | yes (STR) |
| 9 | `dormant_account_high_value_reactivation` | `DORM_AMT_120` + `DORM_KYC_121` | yes |
| 10 | `branchless_banking_level0_breach` | `FLT_CAP_083_tier_cap_exceeded` | no (advisory) |

## Usage — manual ingestion

```bash
# Ingest a sample payload manually
curl -X POST http://localhost:8000/ingest/agents \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/pakistan/agents.json
```

## Usage — running the regression scenarios

```python
import json
import httpx

# Load the 10 test scenarios
with open("tests/fixtures/pakistan/decision_test_scenarios.json") as f:
    suite = json.load(f)

# Get the txn risk scoring scenario UUID for the deployed Pakistan org
# (replace JWT/API_KEY with the Pakistan org's credentials)
r = httpx.get("http://localhost:8080/scenarios", headers={"X-API-Key": API_KEY})
scen_id = next(s["id"] for s in r.json() if s["name"] == "transaction_risk_scoring")

for tc in suite["scenarios"]:
    body = tc["request"]
    body["scenario_id"] = scen_id  # substitute placeholder
    r = httpx.post("http://localhost:8000/decide", json=body, timeout=10)
    decision = r.json()
    expected = tc["expected_outcome"]
    print(f"[{tc['name']:<48}] decision={decision.get('decision', {}).get('outcome')} "
          f"score={decision.get('decision', {}).get('score')} "
          f"escalation={'YES' if decision.get('escalation', {}).get('escalated') else 'no'}")
    # Verify expected_outcome — left as exercise for the deploying team
```

## Pakistan-specific identifier formats used

| Field | Format | Example | Source |
|---|---|---|---|
| `cnic_number` | `XXXXX-XXXXXXX-X` (13 digits) | `42101-1234567-1` | NADRA |
| `iban` | `PK + 2 check + 4 bank + 16 acct` (24 chars) | `PK36SCBL0000001123456702` | SBP IBAN spec |
| `mobile_number` | `+92 3XX XXXXXXX` | `+923001234567` | PTA |
| `ntn_number` | `NNNNNNN-N` (7 digits + check) | `1234567-2` | FBR |
| `secp_registration_no` | `SECP-NPO-YYYY-XXXX` | `SECP-NPO-2018-1234` | SECP |
| `district_code` / `province_code` | `<PROV>-<NN>` / 2-3 letter | `PB-01`, `SD`, `KP` | This pack's seed CSVs |

## Licensing note

These fixtures contain only synthetic data. The CNIC numbers, NTN, IBANs, names, and addresses are generic placeholders and do not correspond to real individuals or entities. Real bank deployments should populate fixtures with their own test data per their data-handling policies.

## To build fixtures for a different bank in Pakistan

Copy this directory to a sibling under your bank's pack name and adjust the values:

```bash
cp -r tests/fixtures/pakistan tests/fixtures/<yourbank>_pakistan
```

Then update the bank-specific fields (operator name, branch codes, address details, etc.) while keeping the regulatory-format identifiers (CNIC, NTN, IBAN, mobile) consistent with NADRA/FBR/SBP/PTA specifications.
