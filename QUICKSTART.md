# Quickstart — Clone to Running in ~10 Minutes

> For someone who just `git clone`d this repo and has zero Marble background. By the end of this you'll have all 9 services running locally, an admin login to the Marble UI, 83 fraud-detection rules loaded, and a working `/decide` endpoint that escalates risky transactions through Checkmarble → Marbel (XGBoost) → GNN.

## What this product is, in one paragraph

**Marble FDS** is a fraud-detection backend for banks and electronic money institutions (EMIs). It's a **generic platform** (the `bridge/` service, the `scripts/setup.py` provisioner, and the `docker-compose.yml`) plus a **jurisdiction pack** that supplies country-specific rules and seed data. The reference pack shipped in this repo is **`pakistan`** — a ~85-rule template covering SBP / FMU / FATF compliance, built entirely from public regulatory documents. To deploy for a different country (Kenya, Indonesia, or a bank-specific tightening of Pakistan…), you copy that pack as a template and edit the rules; the platform itself doesn't change. Three scoring layers run in series for every decision: Checkmarble rules (Layer 1), Marbel XGBoost behavioral model (Layer 2, only if Layer 1 says ≥40), HybridGNN graph model (Layer 3, only if Layer 2 says ≥60). Everything is suggestive — final calls go to a Compliance Officer.

---

## 1. Prerequisites

| Tool | Why | How |
|------|-----|-----|
| **Docker + Docker Compose** | Brings up the 9 services | Linux/Mac: native Docker Engine. Windows: **Docker Engine on WSL2** is recommended (Docker Desktop is known to break VS Code's WSL integration in some setups). |
| **Python 3.10+** | Provisioning scripts + tests + ML training | Most modern Linux/Mac/WSL distros ship with this |
| **`pip`** | Install pytest/httpx/requests for tests | `python3 -m pip install --user pytest httpx requests` (Ubuntu 24+: add `--break-system-packages` per PEP 668) |
| **NVIDIA GPU + CUDA** *(optional)* | GNN retraining only. For inference, CPU works fine. | Only needed if you want to retrain models from scratch |
| **Disk space** | ~5 GB for images + ~500 MB if you train | — |
| **RAM** | 8 GB minimum, 16 GB recommended | — |

---

## 2. Configure your environment

```bash
cd Marble-main
cp .env.example .env
```

Edit `.env`. The fields you **must** change before first start:

| Field | What to set | Example |
|-------|-------------|---------|
| `PG_PASSWORD` | Strong password for Postgres | `openssl rand -hex 24` |
| `JWT_SIGNING_KEY` | Bridge JWT signing key | `openssl rand -hex 32` |
| `SESSION_SECRET` | Frontend session secret (≥32 chars) | `openssl rand -hex 32` |
| `CREATE_ORG_NAME` | Your operator name (shows in UI) | `MyBank Pakistan` |
| `CREATE_ORG_ADMIN_EMAIL` | First admin user's email | `admin@mybank.com` |
| `ADMIN_PASSWORD` | First admin user's password | (your choice; required by scripts) |

Leave `CHECKMARBLE_API_KEY` blank for now — you'll generate it in step 5 after first boot.

`PG_PORT=55432` is the default in the example for a reason: Ubuntu's system PostgreSQL listens on 5432 and would conflict. Don't change unless you know the host port is free.

### Firebase auth: real or emulator?

The repo defaults to **real Firebase** (`docker-compose.yml` references `FIREBASE_API_KEY`, `FIREBASE_PROJECT_ID`, `FIREBASE_AUTH_DOMAIN` from `.env`, plus mounts a service account JSON). Two paths:

- **(A) Real Firebase** (recommended for anything but throwaway demos): create a free Firebase project at https://console.firebase.google.com, enable Email/Password auth, paste `apiKey` / `projectId` / `authDomain` into `.env`, then download a service account JSON (Project Settings → Service accounts → Generate new private key) and save it to `secrets/firebase-service-account.json` (gitignored).
- **(B) Emulator** (pure-local, zero external setup, but state is in-memory): edit `docker-compose.yml` — uncomment the two `FIREBASE_AUTH_EMULATOR_HOST: firebase-auth:9099` lines, replace the env-var refs with `placeholder`/`test-project`, drop the `GOOGLE_APPLICATION_CREDENTIALS` line. Leave `FIREBASE_API_KEY` etc. unset in `.env`.

Step 4 below has both paths.

---

## 3. Bring the stack up

```bash
docker compose up -d --build     # first time: pulls + builds, ~10–20 min
docker compose ps                 # verify all 9 are 'Up' / 'healthy'
```

Daily restarts are seconds:
```bash
docker compose up -d              # no rebuild
docker compose stop               # graceful shutdown, preserves data
docker compose down               # tears down containers (preserves named volume)
docker compose down -v            # NUKES the database volume; admin user is wiped
```

---

## 4. Create the admin user in Firebase

The Marble backend's database gets bootstrapped automatically with your `CREATE_ORG_ADMIN_EMAIL` — but Firebase (where passwords live) doesn't know about that user yet. Bridge them.

**This repo is configured for real Firebase by default.** Pick the path that matches your setup.

### (A) Real Firebase (recommended — what's in `.env` if you copied from `.env.example` and filled in `FIREBASE_*` values)

Prereqs: a Firebase project with Email/Password auth enabled, the web SDK config in `.env` (`FIREBASE_API_KEY`, `FIREBASE_PROJECT_ID`, `FIREBASE_AUTH_DOMAIN`), and the Admin SDK service account JSON saved at `secrets/firebase-service-account.json` (gitignored).

```bash
EMAIL="${CREATE_ORG_ADMIN_EMAIL:-admin@example.com}"
PASSWORD="${ADMIN_PASSWORD:?set ADMIN_PASSWORD in .env first}"
API_KEY="${FIREBASE_API_KEY:?set FIREBASE_API_KEY in .env first}"
PROJECT_ID="${FIREBASE_PROJECT_ID:?set FIREBASE_PROJECT_ID in .env}"

# 1. signUp creates the account in real Firebase
LOCAL_ID=$(curl -s -X POST \
  "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\",\"returnSecureToken\":true}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['localId'])")
echo "Created localId: $LOCAL_ID"

# 2. Mark emailVerified=true via Admin SDK (real Firebase rejects unverified logins)
python3 - <<PY
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import httpx
creds = service_account.Credentials.from_service_account_file(
    "secrets/firebase-service-account.json",
    scopes=["https://www.googleapis.com/auth/identitytoolkit",
            "https://www.googleapis.com/auth/cloud-platform"],
)
creds.refresh(Request())
r = httpx.post(
    f"https://identitytoolkit.googleapis.com/v1/projects/${PROJECT_ID}/accounts:update",
    headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
    json={"localId": "${LOCAL_ID}", "emailVerified": True}, timeout=15,
)
print("verify:", r.status_code, r.json().get("emailVerified"))
PY
```

> **Why two steps:** real Firebase requires email verification before sign-in. The Admin SDK lets us flip the flag without sending the user an email. The service account JSON is the credential that authorizes that — `pip install google-auth httpx` if not already present.

### (B) Firebase emulator (rollback / pure-local dev)

If you've reverted `docker-compose.yml` to emulator mode (uncomment `FIREBASE_AUTH_EMULATOR_HOST: firebase-auth:9099`, drop `GOOGLE_APPLICATION_CREDENTIALS`):

```bash
EMAIL="${CREATE_ORG_ADMIN_EMAIL:-admin@example.com}"
PASSWORD="${ADMIN_PASSWORD:?set ADMIN_PASSWORD in .env first}"
python3 - <<PY
import httpx
fb = "http://localhost:9099"
r = httpx.post(f"{fb}/identitytoolkit.googleapis.com/v1/accounts:signUp?key=placeholder",
               json={"email": "${EMAIL}", "password": "${PASSWORD}", "returnSecureToken": True}, timeout=10)
local_id = r.json()["localId"]
httpx.post(f"{fb}/identitytoolkit.googleapis.com/v1/projects/test-project/accounts:update",
           json={"localId": local_id, "emailVerified": True},
           headers={"Authorization": "Bearer owner"}, timeout=10)
print("done")
PY
```

> Emulator state is in memory — every `firebase_auth` container restart wipes the user. Re-run the snippet if logins start failing with `EMAIL_NOT_FOUND`.

You can now log in at **http://localhost:3000** with the email + password.

---

## 5. Get a Marble API key

The bridge talks to Marble using an API key issued by Marble itself (random hex won't work — the API only accepts keys it created).

In the UI: top-right menu → **Settings → API keys → Create new key** (set role = `API_CLIENT`).
Copy the key value, paste it into `.env`:

```bash
CHECKMARBLE_API_KEY=<the-key-value-you-just-got>
```

Then restart the bridge so it picks up the change:

```bash
docker compose up -d bridge       # NOTE: 'restart' does NOT re-read .env, 'up -d' does
```

---

## 6. Provision the platform (12 tables, 15 scenario shells, 12 list containers)

```bash
CREATE_ORG_ADMIN_EMAIL="$CREATE_ORG_ADMIN_EMAIL" \
ADMIN_PASSWORD="$ADMIN_PASSWORD" \
python3 scripts/setup.py
```

This is **jurisdiction-neutral** — empty schemas, no rules, no seed data. Takes ~30 seconds. After this, you can either:

**(a) Load the Pakistan reference pack** (~85 rules + 12 seeded lists):
```bash
ADMIN_PASSWORD="$ADMIN_PASSWORD" \
python3 scripts/rules_library/pakistan.py
```

**(b) Build your own pack** for your country: copy `scripts/rules_library/pakistan.py` to `scripts/rules_library/<your_pack>.py`, edit the `RULES_BY_SCENARIO` dict, drop seed CSVs in `seed_data/<your_pack>/`, then run it.

**(c) Use the UI**: log in at http://localhost:3000 and add rules to each scenario interactively.

---

## 7. Verify it works

```bash
# Health checks — all 4 must say healthy/ok
curl http://localhost:8000/health            # Bridge
curl http://localhost:5000/health            # Marbel (XGBoost)
curl http://localhost:5001/health            # GNN (HybridGNN v4)
curl http://localhost:8080/health            # Checkmarble API

# Run the test suite (33 tests in ~5 seconds)
# Real-Firebase mode (default — uses values from .env):
set -a; source .env; set +a
CHECKMARBLE_API_URL=http://localhost:8080 BRIDGE_URL=http://localhost:8000 \
FIREBASE_URL=https://identitytoolkit.googleapis.com FIREBASE_API_KEY="$FIREBASE_API_KEY" \
CREATE_ORG_ADMIN_EMAIL="$CREATE_ORG_ADMIN_EMAIL" ADMIN_PASSWORD="$ADMIN_PASSWORD" \
python3 -m pytest tests/ -v

# Emulator mode (only if you've rolled back to the emulator):
# FIREBASE_URL=http://localhost:9099 python3 -m pytest tests/ -v

# Optional: banking-grade reliability check (idempotency, latency, concurrency, error model, three-tier flow)
FIREBASE_URL=https://identitytoolkit.googleapis.com FIREBASE_API_KEY="$FIREBASE_API_KEY" \
  python3 tests/banking_grade_check.py
```

You should see **33 passed** from pytest and **6 PASS / 0 FAIL** from `banking_grade_check.py`.

---

## 8. (Optional) Retrain the ML models

Marbel and GNN ship with pre-trained weights — you don't have to retrain. But if you want to:

```bash
# Both training scripts need PaySim. Get it from Kaggle (~1 min on a fast pipe).
# Either set KAGGLE_API_TOKEN in your env, or place ~/.kaggle/kaggle.json with a real key.
mkdir -p ~/.kaggle
echo '{"api_token":"<your-token>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip3 install --user kaggle
~/.local/bin/kaggle datasets download -d ealaxi/paysim1 --unzip \
  -p bridge/training/data/paysim/

# Marbel (XGBoost) — fast on CPU, ~5 min
python3 bridge/training/scripts/train_marbel_v2.py

# GNN (HybridGNN v4) — needs CUDA-capable GPU; ~3 min on RTX 4060 Ti, much longer on CPU
python3 bridge/training/scripts/cuda_warmup.py        # sanity-check GPU
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 bridge/training/scripts/train_gnn_v4_paysim.py
```

After GNN training: artifacts auto-save to `bridge/gnn/artifacts/`. Restart the GNN container so it picks up the new weights:
```bash
docker compose build gnn && docker compose up -d gnn
```

---

## 9. Make a test decision via the API

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "<paste-scenario-id-from-/scenarios>",
    "trigger_object_type": "transactions",
    "trigger_object": {
      "object_id": "test-001",
      "transaction_id": "test-001",
      "from_wallet_uid": "user-1",
      "to_wallet_uid": "merchant-1",
      "amount": 100000,
      "currency": "AFN",
      "transaction_type": "cash_out",
      "transaction_status": "Success",
      "timestamp": "2026-04-27T22:00:00Z",
      "geo_location": "34.5,69.2",
      "imei": "351111111111111",
      "channel": "App",
      "initiated_by": "user-1",
      "reversal_flag": false,
      "velocity_spike_detected": true,
      "is_night_tx": true,
      "imei_mismatch_flag": false,
      "biometric_failure_count": 0
    }
  }'
```

You'll get back a JSON document with: `decision` (Marble's rules-based score), `escalation` (Marbel + GNN combined score, risk_level, suggestion), and `compliance` (STR/CTR draft signals).

---

## Common gotchas

| Symptom | Cause | Fix |
|---------|-------|-----|
| `port 5432 already allocated` | Host Postgres is running | Keep `PG_PORT=55432` in `.env` |
| `docker: command not found` in fresh shell after install | Group membership not yet applied | New shell, or use `sg docker -c "..."` |
| Tests skip with `Cannot connect to Marble API` | Admin user missing from Firebase, or `FIREBASE_URL` mismatch | Re-run step 4. Real-Firebase tests need `FIREBASE_URL=https://identitytoolkit.googleapis.com` + `FIREBASE_API_KEY="$FIREBASE_API_KEY"` |
| Bridge ignores `.env` changes | `restart` doesn't re-read env | `docker compose up -d bridge` (not `restart`) |
| `EMAIL_NOT_FOUND` from Firebase | Admin user not in Firebase | Re-run step 4 (real or emulator path) |
| Backend panics: `error getting Auth client: google: could not find default credentials` | Real Firebase mode without service account | Save the Firebase Admin SDK JSON to `secrets/firebase-service-account.json` (gitignored); ensure `GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/firebase-sa.json` is in compose |
| Login returns 401 with `email not verified` | New user from `signUp` doesn't have `emailVerified=true` (real Firebase only) | Run the Admin SDK `accounts:update` snippet from step 4(A) part 2 |
| `firebase_auth` container keeps coming back up | `app` declares it as `depends_on` | After `up -d`, run `docker compose stop firebase_auth` (real-Firebase mode doesn't use it) |

---

## What's next?

| Goal | Where to look |
|------|---------------|
| Understand the data flow end-to-end | `SYSTEM_FLOW.md`, `spec.md` |
| Add rules / change thresholds | `scripts/rules_library/<pack>.py` |
| Add a new jurisdiction (Kenya, Indonesia…) | Copy `scripts/rules_library/pakistan.py` and edit; see `docs/jurisdictions/pakistan/README.md` for the pattern |
| Production hardening (auth, HMAC, error sanitization) | `SECURITY.md` + the env vars in `.env.example` |

Welcome to the project.
