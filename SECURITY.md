# Security — Secrets Management & Rotation

> This file documents which secrets the Marble FDS platform uses, where they live, and how to rotate them without downtime. **Never commit real secret values to git.**

---

## Secret inventory

| Secret | Env var | Scope | Blast radius |
|--------|---------|-------|--------------|
| PostgreSQL superuser password | `PG_PASSWORD` | `db`, `api`, `cron`, `decay` containers | All Checkmarble data |
| Checkmarble JWT signing key (RSA 2048 private) | `JWT_SIGNING_KEY` | `api`, `cron` | All user sessions, API auth |
| Checkmarble REST API key | `CHECKMARBLE_API_KEY` | `bridge`, `decay`, tests, scripts | All Checkmarble admin ops |
| Session cookie secret (frontend) | `SESSION_SECRET` | `app` | Frontend sessions |
| Firebase emulator placeholder | `FIREBASE_API_KEY` | `api`, `app`, `firebase_auth` | Dev-only; never real Firebase |
| Marble license key | `LICENSE_KEY` | `api`, `cron` | Dev placeholder |

All are loaded from `.env` at container start. **`.env` is gitignored.**

---

## Where secrets may NOT live

- ❌ Not in `*.py` source files
- ❌ Not in `*.md` documentation
- ❌ Not in `docker-compose.yml` (use `${VAR}` interpolation only)
- ❌ Not in test files (use `os.getenv("VAR", "")`; tests skip if unset)
- ❌ Not in `SECRETS.env` if committed (this file is gitignored but exists on dev machines only as a convenience copy of `.env`)
- ❌ Not in `SETUP_GUIDE.md` values section (the current guide uses `<placeholders>`)

If you run `grep -rn "MIIE\|BEGIN PRIVATE\|[0-9a-f]\{60,\}" --include="*.py" --include="*.md"` from the repo root and see any hits outside of `.env.example` or `tests/sample_payloads/`, **that's a leak** — fix it immediately.

---

## Generate fresh secrets

```bash
# PostgreSQL password (hex-16, 128-bit)
openssl rand -hex 16

# Session cookie secret (hex-20, 160-bit)
openssl rand -hex 20

# JWT signing key (RSA 2048) — output has \n escapes baked in for .env
openssl genrsa 2048 | sed ':a;N;$!ba;s/\n/\\n/g'

# Checkmarble API key: generate via Admin UI → Settings → API Keys → Create
```

---

## Rotation procedure (zero downtime preferred)

### A. Checkmarble API key

Lowest blast radius. Rotate first.

```bash
# 1. In Admin UI (http://localhost:3000), create a new API key. Copy it.
# 2. Update .env:
sed -i "s/^CHECKMARBLE_API_KEY=.*/CHECKMARBLE_API_KEY=<new-key>/" .env
# 3. Bounce only the services that read it (no DB downtime):
docker compose up -d --no-deps bridge decay
# 4. Verify:
curl -s http://localhost:8000/health | jq .checkmarble_connected
# 5. Back in Admin UI, revoke the old key.
```

### B. PostgreSQL password (rolling)

Needs a brief reconnect on each consumer. Use dual-password window:

```bash
# 1. Connect to Postgres directly and set new password WITHOUT dropping the old one:
docker exec -it marble-postgres psql -U postgres -d marble \
  -c "ALTER USER postgres WITH PASSWORD '<NEW>';"
# Postgres replaces the hash atomically; any new connection picks up the new password.
# 2. Update .env with the new password.
# 3. Recreate the containers that hold PG connection pools so they reconnect:
docker compose up -d --force-recreate --no-deps api cron bridge decay
# 4. Verify:
docker compose logs --tail 20 api | grep -i "connected\|error"
```

> Production note: switch to **two distinct roles** (`app_blue`, `app_green`) each with its own password, + pgBouncer with `auth_query`, to do this with zero connection loss. For dev/pitch demo the simpler approach above is fine.

### C. JWT signing key (zero-downtime via JWKS/kid)

The current Checkmarble API uses a single signing key from env, so rotation = restart with the new key (tokens issued before rotation become invalid). For a true zero-downtime rotation, adopt JWKS:

1. Store keys as a list in a JWKS JSON, each with a unique `kid` (key ID)
2. Sign new tokens with the newest `kid`
3. Verify tokens against any `kid` still in the list
4. After the max token TTL has elapsed, remove the old `kid`

Recommended library: `authlib>=1.3` or `python-jose[cryptography]>=3.3`. This requires a patch to the Checkmarble backend (upstream v0.59 doesn't support JWKS out of the box); alternatively, gate behind short-TTL access tokens + refresh tokens.

For the immediate pitch:

```bash
# 1. Generate new key (escaped-newline form for .env)
openssl genrsa 2048 | sed ':a;N;$!ba;s/\n/\\n/g' > /tmp/new_jwt.txt
# 2. Update .env JWT_SIGNING_KEY= with the new value
# 3. Restart auth-bearing containers. All existing session tokens will be invalidated:
docker compose up -d --force-recreate --no-deps api cron app
# 4. Users must re-authenticate.
```

### D. Session cookie secret

```bash
# 1. Update .env SESSION_SECRET=<new>
# 2. Recreate frontend:
docker compose up -d --force-recreate --no-deps app
# All existing sessions invalidated; users re-login.
```

---

## Pre-flight checklist before a pitch or hearing

- [ ] `.env` file exists locally, values populated
- [ ] `grep -r "93517c4964\|c9dc14bfbd\|MIIEvwIBAD" .` returns zero hits (old dev secrets)
- [ ] `git status` shows `.env` + `SECRETS.env` as untracked (gitignored)
- [ ] `docker compose config --quiet` passes (no interpolation errors)
- [ ] `docker compose up -d` — all services healthy within 60s
- [ ] `curl http://localhost:8000/health` returns `checkmarble_connected: true`
- [ ] `pytest tests/ -v` passes
- [ ] If rotating for a fresh demo: rotate Checkmarble API key, regenerate JWT, restart stack

---

## Incident response — suspected key compromise

1. **Rotate immediately** per sections A–D above (in that order — API key first, PG last).
2. **Audit access**: `docker exec marble-postgres psql -U postgres -d marble -c "SELECT * FROM audit_trail WHERE timestamp > NOW() - INTERVAL '24 hours' ORDER BY timestamp DESC"`
3. **Revoke sessions**: restart `api` + `app` (invalidates all JWT tokens).
4. **Check git history** for the compromised value: `git log --all -p | grep -F "<leaked-value>"`. If found in a commit, rewrite history (`git filter-repo`) and force-push; all collaborators must re-clone.
5. Document the incident in a private `INCIDENT_<date>.md` — never commit it.

---

## Known legacy secrets (INVALIDATE before pitch)

These values appeared in prior commits / documentation and should be treated as compromised. **All have been removed from tracked files as of 2026-04-21** but may still exist in the local `.env` / `SECRETS.env` on dev machines:

- `PG_PASSWORD=c9dc14bfbd9fea34345077cd635c1022` — rotate
- `CHECKMARBLE_API_KEY=93517c4964a64160422471f3439132554691217e4f75ba62260a33621f6cee80` — revoke via Admin UI
- `SESSION_SECRET=a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9` — rotate
- The JWT RSA private key previously in `SETUP_GUIDE.md:205` — rotate

After rotation, verify none of these values appear anywhere in the repo:

```bash
grep -rn "c9dc14bfbd9fea34345077cd635c1022\|93517c4964a64160422471f3439132554691217e4f75ba62260a33621f6cee80\|a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9\|MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkw" . 2>/dev/null
```

Should return no hits.
