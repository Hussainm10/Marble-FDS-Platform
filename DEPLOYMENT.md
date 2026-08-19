# Deployment Guide

> Two big questions answered here:
>
> 1. **"It runs on localhost — how do I actually deploy this somewhere people can use it?"**
> 2. **"Marble has a pricing page on their website. Do I have to pay them anything?"**
>
> Short answers: yes you can deploy it (lots of options, walked through below), and no, you don't have to pay anything to Marble for self-hosting — but there's a license nuance worth knowing.

---

## Part 1 — Marble's licensing & billing (the short version)

You're using **Marble's open-source backend** (https://github.com/checkmarble/marble-backend) under the **Elastic License 2.0**. What that means in plain English:

| Thing you can do | Allowed? |
|---|---|
| Run Marble for your own bank or EMI, free, forever | ✅ Yes |
| Modify the source code | ✅ Yes |
| Use it commercially (your bank earns money using it) | ✅ Yes |
| Self-host on your own servers / cloud | ✅ Yes |
| Wrap Marble and sell it as a managed SaaS to other banks | ❌ No (this is the only real restriction) |
| Remove Marble's copyright/license notices | ❌ No |

**You owe Marble nothing for self-hosting.** The Docker images we pull from `europe-west1-docker.pkg.dev/marble-infra/...` are public — no auth, no rate limits, no usage fees.

### What the pricing page on marble.com is for

Marble offers three tiers commercially:

| Tier | Price | What you get | Your situation |
|---|---|---|---|
| **Free / Open Source** | $0 | Self-hosted, full feature set, community support | ← **This is what we're using** |
| **Starter** | Contact sales | Marble Cloud (managed SaaS, they run it for you) + investigation tools + direct support | Optional. Costs money. |
| **Enterprise** | Contact sales | SaaS or on-prem + AI agents + SLA + EBA-compliant contracting | Optional. Costs money. |

**Self-hosted == free, forever.** The paid tiers exist if you don't want to manage your own infrastructure. They're functionally a hosting service.

### What this project adds on top of Marble

Everything in this repo *outside* the pulled Docker images is yours:

- `bridge/` — our FastAPI service (idempotency, auth, error model, structlog, Prometheus, escalation, compliance)
- `bridge/marbel/` — XGBoost model + service
- `bridge/gnn/` + `bridge/gnn_stub/` — GNN model + service
- `scripts/` — provisioning + jurisdiction packs
- `seed_data/` — list contents
- `tests/` — 33 pytest + banking-grade reliability check

You wrote this. No external license applies (other than the Python libs, all permissive).

---

## Part 2 — Deployment options

The fastest path: take what we have working locally and run the same Docker Compose on a Linux VM somewhere. Then progressively harden.

### Option A — Single VM, basic (good for dev/staging or a pilot bank)

**You need:** any Linux VM with ≥4 vCPU, ≥8 GB RAM, ≥40 GB disk. Examples:
- AWS EC2 `t3.large` or `c6i.xlarge`
- GCP Compute `e2-standard-4`
- Azure `Standard_B4ms`
- DigitalOcean Droplet 8 GB / 4 vCPU
- Hetzner CX31 (~€7/mo, perfectly fine for staging)

**Steps (works on any of those):**

```bash
# 1. SSH in, install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# 2. Clone your repo
git clone https://github.com/<you>/Marble-main.git
cd Marble-main

# 3. Set up .env (DIFFERENT secrets from your laptop — generate fresh)
cp .env.example .env
# Edit: generate NEW PG_PASSWORD, JWT_SIGNING_KEY, SESSION_SECRET via openssl
# Edit: set CREATE_ORG_NAME, CREATE_ORG_ADMIN_EMAIL, ADMIN_PASSWORD

# 4. First boot
docker compose up -d --build
# Wait ~5 min for everything to be healthy. Then follow QUICKSTART.md
# steps 4 (Firebase bootstrap) → 5 (API key) → 6 (provision) onwards.

# 5. Open the bridge to the internet
# (Skip this if you only want internal use — keep it on a VPN.)
sudo ufw allow 8000/tcp     # bridge
sudo ufw allow 3000/tcp     # admin UI
sudo ufw enable
```

**You now have a working server.** Hit `https://<vm-ip>:8000/decide` from anywhere.

**Caveats for this option:**
- No HTTPS yet (see Option B for TLS).
- No backups yet (see "Backups" section).
- Single point of failure (one VM dies → outage). Fine for dev/staging, not for prod payments.

### Option B — Single VM with TLS + reverse proxy (good for production pilot)

Add nginx (or Caddy, or Traefik) in front of the bridge for TLS termination. The simplest path is **Caddy** because it auto-issues Let's Encrypt certs:

```bash
# Install Caddy on the VM
sudo apt install -y caddy

# /etc/caddy/Caddyfile
api.yourbank.com {
    reverse_proxy localhost:8000
}
admin.yourbank.com {
    reverse_proxy localhost:3000
}

sudo systemctl reload caddy
```

Point the DNS records `api.yourbank.com` and `admin.yourbank.com` at your VM's IP, and Caddy gets you HTTPS automatically. Now bridge calls go to `https://api.yourbank.com/decide`.

**Recommended add-ons at this stage:**
- **Set `BRIDGE_API_KEY`** in `.env` to require `X-Bridge-API-Key` on every request (currently optional in dev).
- **Set `WEBHOOK_SECRET`** to verify HMAC on incoming webhooks.
- **Set `DEBUG_DETAILS=false`** (or unset — it defaults to off) so error responses don't leak internals.

### Option C — Container orchestration (Kubernetes, ECS, Cloud Run)

For real production with horizontal scale, multi-AZ, zero-downtime deploys, etc.:

| Platform | What you'd do |
|---|---|
| **AWS ECS Fargate** | Push images to ECR, define a task per service in `docker-compose.yml`, put an ALB in front of bridge + admin |
| **AWS EKS / GCP GKE / Azure AKS** | Build/push images, write Helm chart or plain manifests (one Deployment + Service per docker-compose service), use a managed Postgres (RDS / Cloud SQL) instead of the local container |
| **Google Cloud Run** | Per-service Cloud Run deployments — bridge and ML services. Use Cloud SQL for Postgres. The Marble backend itself runs on Cloud Run too. |
| **Marble's own deployment guide** | Marble's docs at https://docs.checkmarble.com (the public docs site) include their reference architectures for K8s and Cloud Run |

The thing that changes from Option A → C is mostly:
- **Postgres**: stop running it as a container; use a managed one (RDS, Cloud SQL, Crunchy, Neon).
- **Firebase**: stop using the emulator; create a real Firebase project (their free Spark tier is plenty for auth) and point Marble at it.
- **Object storage**: set the `INGESTION_BUCKET_URL`, `CASE_MANAGER_BUCKET_URL`, `ANALYTICS_BUCKET_URL` to real S3/GCS buckets so investigation evidence and case files persist.
- **Secrets**: stop using `.env` files; use Secrets Manager / Secret Manager / Key Vault.

Same Docker images, same code — just plumbed through proper cloud primitives.

### Option D — Marble Cloud (the paid SaaS)

If you want **zero infra work**, you can:

1. Sign up at https://www.checkmarble.com/contact for a Marble Cloud trial.
2. Re-host **only our additions** (bridge, marbel, gnn) on a small VM/Cloud Run.
3. Point the bridge at Marble Cloud's API instead of `http://api:8080`.

You'd still self-host the bridge + ML services (because those are ours, not Marble's), but Marble itself becomes a managed dependency. Their pricing isn't public — contact their sales.

---

## What changes between local and prod (concrete checklist)

When you go from local Docker Compose to a real deployment, these are the deltas you'll hit:

| Concern | Local default | Production fix |
|---|---|---|
| **TLS** | None (HTTP) | Reverse proxy with Let's Encrypt (Caddy/Traefik) or cloud LB |
| **Database** | Container `marble-postgres` | Managed Postgres (RDS, Cloud SQL, etc.) — set `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD` env vars on api/cron/bridge |
| **Auth** | Firebase emulator (volatile, in-memory) | Real Firebase project (Auth Spark tier, free) — change Marble's auth config to point at `https://identitytoolkit.googleapis.com` |
| **Bridge auth** | `BRIDGE_API_KEY` unset (open) | Set the env var, distribute the key to clients |
| **Webhook auth** | `WEBHOOK_SECRET` unset (open) | Set it, share the secret with the webhook sender |
| **Error detail** | `DEBUG_DETAILS` defaults off | Keep off; verify error responses say `"internal_service_error"` |
| **Logs** | stdout (visible via `docker logs`) | Ship to CloudWatch / Loki / Datadog. structlog already emits JSON — it'll parse cleanly. |
| **Metrics** | `/metrics` reachable on bridge:8000 | Add Prometheus + Grafana, or ship to a managed metrics platform (Grafana Cloud, Datadog, etc.) |
| **Object storage** | empty bucket URLs | Point at S3 / GCS — Marble uploads case files + analytics dumps there |
| **Backups** | None | Daily logical Postgres dump (`pg_dump`) to S3, point-in-time recovery if your RDS supports it |
| **Image registry** | Public Marble images + local builds | Push your own bridge/marbel/gnn images to ECR / Artifact Registry / GHCR |
| **CI/CD** | manual `docker compose up -d --build` | GitHub Actions: build on push, deploy via `kubectl apply` or Cloud Run revision |

None of these are blockers — they're the standard "go-prod" checklist for any Dockerized app.

---

## "Just clone and run" — does that actually work?

**Yes**, but only on someone's *local* machine, not as a deployable URL. To put it differently:

- A teammate clones the repo → follows `QUICKSTART.md` → has it running on their localhost in ~10 minutes. **Good for development.**
- A teammate clones the repo → wants to give it to their bank's compliance team in another country to test → they need to **deploy** it (Options A–D above). A clone alone doesn't give your team a URL.

So the GitHub clone flow is for **development and onboarding new devs**. Deployment is a separate step on top.

---

## Backups (don't skip this in production)

If you're handling real bank data, you need backups. The minimum:

```bash
# Daily Postgres dump uploaded to S3
docker exec marble-postgres pg_dump -U postgres marble | gzip > marble-$(date +%Y%m%d).sql.gz
aws s3 cp marble-$(date +%Y%m%d).sql.gz s3://yourbank-marble-backups/
```

Run that as a cron / systemd timer / Cloud Scheduler job. Test restore monthly.

For the bridge / ML services, backups don't matter — they're stateless. The artifacts in `bridge/gnn/artifacts/` and `bridge/marbel/artifacts/` are checked into git, so they're already backed up.

---

## Summary

- **Marble itself: free** under Elastic License 2.0 if you self-host. The only thing you can't do is repackage it as a SaaS for other banks.
- **Quickest production path:** one VM (Hetzner / DO / EC2) → Caddy for TLS → set `BRIDGE_API_KEY` and `WEBHOOK_SECRET` → daily Postgres dumps → done.
- **Real production at scale:** managed Postgres + real Firebase project + container orchestrator (ECS/EKS/Cloud Run) + secrets manager + observability.
- **You don't HAVE to use Marble Cloud** (the paid SaaS) — but it exists as an option if you'd rather pay than manage infra.
- **`git clone` + run is for devs**, not deployment. Deployment is the next step on top.
