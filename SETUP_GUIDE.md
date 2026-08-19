# Marble FDS — Setup Guide

> Setup instructions for the generic Marble FDS platform on a fresh machine. Follow each step exactly. For jurisdiction-specific rule packs (e.g., the Pakistan reference pack), see `scripts/rules_library/` after completing platform setup.

---

## Quick Overview

This is a **jurisdiction-agnostic Fraud Detection System (FDS)** with three-tier scoring:

1. **Checkmarble** — rule engine (scenarios + rules added by the operator's jurisdiction pack or via Admin UI)
2. **Marbel** — XGBoost behavioral model (AUC-ROC 0.9994, trained on PaySim)
3. **GNN** — Graph Neural Network for entity analysis

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.12+ | `python3 --version` |
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Git | 2.30+ | `git --version` |

### Install Prerequisites (Ubuntu/Debian/WSL2)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3.12 python3.12-venv python3.12-dev -y
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
sudo apt install docker-compose-plugin -y
```

---

## Step 1: Clone Repository

```bash
git clone <your-repo-url> Marble-main
cd Marble-main
ls -la   # Should see: bridge/, scripts/, tests/, docker-compose.yml, etc.
```

---

## Step 2: Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
which python   # /path/to/Marble-main/.venv/bin/python
```

---

## Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r bridge/requirements.txt
# Optional — only if you plan to retrain ML models:
pip install -r bridge/training/requirements-train.txt
```

### Explicit dependency list (if requirements.txt fails)

```bash
# Core API
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 httpx==0.28.1 pydantic==2.10.5

# ML Core
pip install xgboost==2.1.3 scikit-learn==1.5.2 imbalanced-learn==0.12.4 lightgbm==4.5.0

# Deep Learning (GNN)
pip install torch==2.4.1 torch-geometric==2.6.1

# Explainability & Data
pip install shap==0.46.0 pandas==2.2.3 numpy==1.26.4

# Utilities
pip install optuna==4.0.0 kaggle==1.6.17 networkx==3.4.2 joblib==1.4.2 tqdm==4.66.5 python-dotenv==1.0.1
```

---

## Step 4: Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your values (see SECURITY.md for rotation + generation)
```

### Generate fresh secrets

```bash
openssl rand -hex 16                            # PG_PASSWORD
openssl rand -hex 20                            # SESSION_SECRET
openssl genrsa 2048 | sed ':a;N;$!ba;s/\n/\\n/g' # JWT_SIGNING_KEY
```

Obtain a `CHECKMARBLE_API_KEY` from the Admin UI at `http://localhost:3000` → Settings → API Keys after the stack is running.

See `SECURITY.md` for full rotation procedures.

---

## Step 5: Download Training Datasets (optional)

Only needed if you plan to retrain ML models. Training data is **not in the git repo** — download from Kaggle.

### Option A — automated script

```bash
source .venv/bin/activate
python scripts/download_training_data.py
```

Requires Kaggle credentials — see Step 5.1 below.

### Step 5.1 — Kaggle credentials

1. Create account at https://www.kaggle.com/
2. Go to https://www.kaggle.com/settings → API → Create New Token (downloads `kaggle.json`)
3. Install:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets list   # should list datasets without error
   ```

### Option B — manual download

- **PaySim** (~471 MB): https://www.kaggle.com/datasets/ealaxi/paysim1 → extract to `bridge/training/data/paysim/`
- **Elliptic** (~666 MB): https://www.kaggle.com/datasets/ellipticco/elliptic-data-set → extract to `bridge/training/data/elliptic/`

### Verify

```bash
ls -lh bridge/training/data/paysim/       # PS_20174392719_1491204439457_log.csv (~471M)
ls -lh bridge/training/data/elliptic/     # features, classes, edgelist CSVs
```

---

## Step 6: Verify Trained Models

Pre-trained model artifacts are committed to the repo. Verify they exist:

```bash
ls -lh bridge/marbel/artifacts/
# Expected:
#   xgboost_v2.json          (554K) — production XGBoost
#   scaler_v2.pkl            (1.5K)
#   calibrator_v2.pkl        (1.7M)
#   shap_explainer_v2.pkl    (3.0M)
#   feature_config_v2.json   (1.9K)

ls -lh bridge/gnn/artifacts/
# Expected:
#   hybrid_v3.pt             (1.6M) — current (weak) GNN
#   scaler_hybrid_v3.pkl     (4.7K)
#   config_hybrid_v3.json    (457B)
```

---

## Step 7: Start Docker Services

```bash
docker compose up -d --build
docker compose logs -f           # watch startup (Ctrl+C to exit)
docker compose ps                # confirm all services running
```

### Expected Services

| Service | Port | Status |
|---------|------|--------|
| marble-postgres | 5432 | Running |
| marble-api | 8080 | Running |
| marble-app | 3000 | Running |
| firebase-auth | 9099 | Running |
| marbel-ml | 5000 | Running |
| gnn-ml | 5001 | Running |
| bridge | 8000 | Running |
| risk-decay-cron | — | Running |

---

## Step 8: Provision the Platform

```bash
export CHECKMARBLE_API_KEY="your-api-key"
./scripts/setup.sh
```

This creates:
- 12 data-model tables with canonical schema
- 12 inter-table links (foreign keys for cross-entity rules)
- 12 empty custom lists (you seed these per your jurisdiction)
- 15 scenario shells with names, triggers, thresholds — **but no rules**

---

## Step 9: Add Rules (pick one)

### Option A — load the Pakistan reference pack

If you're running the reference Pakistan deployment:

```bash
python scripts/rules_library/pakistan.py
```

This adds the 82 canonical the central bank/the jurisdiction's Financial Intelligence Unit (FIU)/AAOIFI rules + seeds the 12 custom lists with Afghan reference data.

### Option B — build your own jurisdiction pack

Copy `scripts/rules_library/pakistan.py` as a template:

```bash
cp scripts/rules_library/pakistan.py \
   scripts/rules_library/my_bank_jurisdiction.py
# Edit to match your AML typologies, KYC tiers, sanctions lists, etc.
python scripts/rules_library/my_bank_jurisdiction.py
```

### Option C — use the Admin UI

Open `http://localhost:3000` → log in → add rules per scenario interactively.

---

## Step 10: Verify the System

### Health checks

```bash
curl http://localhost:5000/health   # Marbel ML
curl http://localhost:5001/health   # GNN ML
curl http://localhost:8000/health   # Bridge
curl http://localhost:8080/liveness # Checkmarble API (expects "OK")
```

### Web dashboard

Open `http://localhost:3000`. Bootstrap admin credentials are set via the `.env` values during first run.

### Run tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
# Or a specific suite:
python -m pytest tests/test_real_models.py -v
```

---

## Retraining Models (optional)

Only needed if you want to retrain from scratch.

### Retrain Marbel (XGBoost)

```bash
source .venv/bin/activate
python bridge/training/scripts/train_marbel_v2.py
```

### Retrain GNN (HybridGNN)

```bash
source .venv/bin/activate
python bridge/training/scripts/train_gnn_v3_hybrid.py
# Or v4 (on PaySim, better mobile-money fit — when available):
# python bridge/training/scripts/train_gnn_v4_paysim.py
```

---

## Troubleshooting

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
# Logout and log in again
```

### Port Already in Use

```bash
sudo lsof -i :8080
# Kill the process or change ports in .env
```

### Database Connection Failed

```bash
docker compose down -v
docker compose up -d
```

### Python Version Issues

```bash
python3 --version
# If < 3.12:
sudo apt install python3.12 python3.12-venv
```

### WSL2 Memory Issues

Create/edit `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
memory=8GB
processors=4
```
Then restart WSL: `wsl --shutdown`

### Models Not Loading

```bash
ls -la bridge/marbel/artifacts/
ls -la bridge/gnn/artifacts/
docker compose logs marbel-ml
docker compose logs gnn-ml
```

---

## Project Structure Reference

```
Marble-main/
├── bridge/                      # Platform services
│   ├── app.py                   # Main Bridge service
│   ├── config.py                # Configuration loader
│   ├── escalation.py            # Three-tier escalation logic
│   ├── compliance.py            # STR/CTR automation
│   ├── marbel/artifacts/        # Trained XGBoost models ✓
│   ├── gnn/artifacts/           # Trained GNN models ✓
│   ├── marbel_stub/             # Marbel FastAPI service
│   ├── gnn_stub/                # GNN FastAPI service
│   ├── decay/                   # Risk decay cron
│   └── training/                # Training scripts
│       ├── scripts/             # train_*.py files
│       └── data/                # Training datasets (download separately)
├── scripts/                     # Setup & utility scripts
│   ├── setup.sh                 # Master orchestrator
│   ├── setup.py                 # Platform provisioner
│   ├── rules_library/           # Jurisdiction rule packs (optional)
│   │   └── pakistan.py
│   ├── _attic/                  # Legacy scripts (retired, kept for reference)
│   └── download_training_data.py
├── seed_data/                   # Jurisdiction-specific reference data
├── tests/                       # Integration + scenario tests
├── docs/                        # Platform + jurisdiction docs
├── docker-compose.yml           # Service orchestration
├── .env.example                 # Environment template
├── .env                         # Your actual config (gitignored)
├── SETUP_GUIDE.md               # This file
├── README.md                    # Project overview
├── spec.md                      # Platform specification
└── SECURITY.md                  # Secrets + rotation
```

---

## Model Performance Summary

### Marbel XGBoost v2
- **AUC-ROC**: 0.9994
- **Precision**: 100%
- **Recall**: 99.8%
- **F1 Score**: 0.9988
- **Dataset**: PaySim (6.3M transactions)
- **Features**: 35 engineered features

### GNN HybridGNN v4
- **AUC-ROC**: 0.990
- **AUC-PR**: 0.891
- **F1 Score**: 0.82
- **Dataset**: PaySim (mobile-money graph, re-trained from the earlier Elliptic-Bitcoin v3 baseline which suffered domain mismatch)

---

## Quick Start Commands

```bash
# Full setup from scratch
git clone <your-repo-url> Marble-main
cd Marble-main
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r bridge/requirements.txt -r bridge/training/requirements-train.txt
cp .env.example .env   # Edit with generated secrets
docker compose up -d --build
./scripts/setup.sh
# optional: python scripts/rules_library/pakistan.py
```

---

**Setup complete!** Your Marble FDS platform should now be running.

For support, check `README.md` or `SECURITY.md`.
