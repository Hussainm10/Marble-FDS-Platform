<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Rules%20%2B%20ML%20%2B%20GNN-orange?style=for-the-badge" alt="Three-tier architecture">
  <img src="https://img.shields.io/badge/Reference%20Pack-Pakistan%20(SBP%2FFMU%2FFATF)-2ea44f?style=for-the-badge" alt="Pakistan reference pack">
  <img src="https://img.shields.io/badge/Stack-FastAPI%20%7C%20XGBoost%20%7C%20PyTorch%20Geometric-blue?style=for-the-badge" alt="Stack">
</p>

<h1 align="center">Marble FDS</h1>
<h3 align="center">A jurisdiction-agnostic fraud detection platform for banks &amp; EMIs</h3>

<p align="center">
  <b>Architect:</b> Hussain Mansoor Bhutto — AI/ML Engineer<br>
  <sub>github.com/Hussainm10 · linkedin.com/in/hussainmansoorbhutto</sub>
</p>

---

## Why This Exists

Most fraud-detection systems banks run today are pure rule engines: fast and explainable, but easy for fraudsters to learn and route around, and blind to coordinated behavior across accounts. Pure ML systems go the other way — they catch novel patterns but are opaque, hard to justify to a regulator, and expensive to retrain per jurisdiction.

**Marble FDS combines both, plus a third layer for organized fraud:** a rules engine for regulatory compliance, a behavioral ML model for anomalies rules can't express, and a graph neural network for mule networks and collusion that neither rules nor a single-transaction ML model can see. Every decision is escalated through these layers only as needed, and every automated decision is *suggestive* — a human Compliance Officer has final authority via a Maker-Checker workflow.

---

## Architecture — Three Tiers, One Decision

```
                    TRANSACTION ARRIVES
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  TIER 1 — RULES (Checkmarble)          │   Weight: 40%
        │  Jurisdiction-pack rules, additive      │
        │  scoring. "Is this legal?"              │
        └───────────────────┬─────────────────────┘
                             │ score ≥ 40 → escalate
                             ▼
        ┌───────────────────────────────────────┐
        │  TIER 2 — ML (XGBoost, "Marbel")        │   Weight: 35%
        │  Behavioral anomaly scoring.             │
        │  "Is this normal for this account?"      │
        └───────────────────┬─────────────────────┘
                             │ score ≥ 60 → escalate
                             ▼
        ┌───────────────────────────────────────┐
        │  TIER 3 — GRAPH NEURAL NETWORK          │   Weight: 25%
        │  Shared devices, linked wallets,         │
        │  circular flows. "Is this part of a      │
        │  ring?"                                  │
        └───────────────────┬─────────────────────┘
                             ▼
              WEIGHTED COMBINED SCORE (0–100)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          APPROVE         REVIEW      BLOCK / DECLINE
                                    → STR/CTR draft
                                    → Compliance Officer queue
                                    → Maker-Checker sign-off
```

Only transactions the rules layer flags as risky ever reach the ML and graph layers — most traffic is scored and approved by Tier 1 alone, keeping latency low.

### What Each Tier Catches

| Tier | Technology | Catches | Example |
|------|------------|---------|---------|
| **Rules** | Score-based rule engine (Checkmarble, open-source) | Known patterns, regulatory breaches | CTR threshold, blacklist hit, velocity limit |
| **ML** | XGBoost, trained on PaySim mobile-money data | Behavioral anomalies | Unusual amount/time/frequency for this account |
| **Graph AI** | Hybrid GNN (PyTorch Geometric) | Organized fraud | Shared device across "unrelated" wallets, circular transfers, mule rings |

---

## The Platform Is Generic — Rules Live in Pluggable "Jurisdiction Packs"

This is the part that makes it more than a single-country demo: the platform itself (the data model, the bridge service, the scoring pipeline, the compliance hooks) ships with **no rules at all**. A deploying bank loads a jurisdiction pack — a Python file mapping their regulator's requirements to scored rules, plus seed CSVs for their blacklists/watchlists — and everything else just works.

```
python scripts/rules_library/pakistan.py      # load the shipped reference pack
# or
cp scripts/rules_library/pakistan.py scripts/rules_library/<your_pack>.py
# edit the rules, drop your own seed CSVs in seed_data/<your_pack>/, run it
```

**Reference pack shipped in this repo: Pakistan.** ~85 rules across 15 scenarios, every threshold traced to a public regulatory source — no bank's internal data or policy went into it:

- State Bank of Pakistan (SBP) AML/CFT/CPF Regulations, Branchless Banking Regulations, Foreign Exchange Manual
- Financial Monitoring Unit (FMU) STR/CTR formats and typology reports
- Anti-Money Laundering Act 2010
- FATF Mutual Evaluation Report + follow-ups
- NADRA (CNIC verification), PTA (blocked-IMEI lists)
- Optional Islamic-banking module (Shariah contract validation, Wakalah delegation) — disabled by default, opt-in for Islamic banks

Full clause-by-clause source mapping: [`docs/jurisdictions/pakistan/REGULATORY_MAPPING.md`](docs/jurisdictions/pakistan/REGULATORY_MAPPING.md).

> **Note on the numbers:** the rule thresholds were compiled via AI-assisted research against secondary sources, not a clause-by-clause read of every current primary regulation. An independent re-check confirmed the core structure and several key figures (FATF grey-list exit date, Branchless Banking Level 0 limits) but flagged a few thresholds as unconfirmed or outdated — see the disclaimer at the top of [`REFERENCES.md`](REFERENCES.md) before treating any specific figure as settled.

Any bank can bring their own Postgres schema, author their own rule pack the same way, and run the identical three-tier pipeline underneath.

---

## Quick Start

```bash
git clone <this-repo-url>
cd Marble-FDS-Platform
cp .env.example .env   # fill in the required secrets — see QUICKSTART.md
./start.sh
```

Full walkthrough with prerequisites, Firebase setup, and troubleshooting: [`QUICKSTART.md`](QUICKSTART.md).

---

## Data Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           12 TABLES | ~280 FIELDS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ENTITIES                   ACTIVITY                  COMPLIANCE           │
│   ────────                   ────────                  ──────────           │
│   individual_users           transactions              risk_events          │
│   merchant_users             notification_logs         audit_trail          │
│   corporate_wallets          biometric_logs             fds_input_features  │
│   agents                     float_delegations                              │
│                              wakalah_delegations                            │
│                                                                             │
│   Canonical field names — designed to sit in front of any core banking DB   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 15 Fraud Scenario Shells (Platform-Level)

| Category | Scenarios | Catches |
|----------|-----------|---------|
| **Identity** | KYC verification, biometric fraud | Fake IDs, spoofing, document fraud |
| **Transaction** | Risk scoring, AML/sanctions screening | Structuring, velocity abuse, sanctions hits |
| **Agent** | Float monitoring, GPS/IMEI compliance | Agent fraud, location spoofing |
| **Corporate** | Bulk payments, delegation compliance | Payroll fraud, unauthorized transfers |
| **Account** | Login anomaly, dormant reactivation | Account takeover, dormant-account abuse |

Scenario triggers and thresholds are platform-defined; the actual rules that fire inside each scenario come from the loaded jurisdiction pack.

---

## Compliance & Governance

- **Maker-Checker workflow** — every automated decision is suggestive; a licensed Compliance Officer has final authority.
- **STR/CTR auto-drafting** — hard-locked to draft-only. The bridge service enforces a hostname allowlist so nothing in this codebase can ever auto-submit to an external regulator; submission requires a human in the loop.
- **Immutable audit trail** with jurisdiction-configurable retention.
- **Explainable at every layer** — a decision always shows which rules fired, the ML feature contribution, and (when escalated) the graph signal that triggered it.

---

## Services (Docker Compose)

| Service | Port | Purpose |
|---------|------|---------|
| Checkmarble Admin UI | [localhost:3000](http://localhost:3000) | Build/edit rules, review decisions |
| Checkmarble API | [localhost:8080](http://localhost:8080) | Rule engine, data ingestion |
| Bridge Service (FastAPI) | [localhost:8000/docs](http://localhost:8000/docs) | Escalation orchestration, webhooks, compliance |
| Marbel ML | localhost:5000 | XGBoost behavioral scoring |
| GNN ML | localhost:5001 | Graph-based fraud analysis |
| PostgreSQL | localhost:5432 | Platform data store |

---

## Documentation

| Document | For |
|----------|-----|
| [`spec.md`](spec.md) | Full platform specification |
| [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md) | Step-by-step request/decision flow |
| [`QUICKSTART.md`](QUICKSTART.md) | Developers — clone to running |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | DevOps / production hardening |
| [`SECURITY.md`](SECURITY.md) | Security review |
| [`REFERENCES.md`](REFERENCES.md) | Compliance officers — every regulatory source cited |
| [`docs/jurisdictions/pakistan/`](docs/jurisdictions/pakistan/) | The Pakistan reference pack, in depth |

---

## About

Built by **Hussain Mansoor Bhutto**, AI/ML Engineer, as an exploration of how rule-based compliance engines, behavioral ML, and graph neural networks combine into one fraud-detection pipeline for the financial sector. Open to conversations with banks, EMIs, and fintechs interested in this architecture.

- GitHub: [github.com/Hussainm10](https://github.com/Hussainm10)
- LinkedIn: [linkedin.com/in/hussainmansoorbhutto](https://linkedin.com/in/hussainmansoorbhutto)
