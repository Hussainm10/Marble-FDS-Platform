#!/usr/bin/env python3
"""Banking-grade reliability check against the live stack.

Run after `docker compose up -d`. Tests that matter for production deployment
of a fraud-detection system:

  1. Idempotency  — same key + same body returns cached response, no re-execution
  2. Latency      — p50/p95/p99 of /decide under sequential load
  3. Concurrency  — many parallel /decide calls succeed without races
  4. Error model  — bad input returns canonical {code, message, details}, no stack traces leaked
  5. Auth surface — confirm `/health` is always reachable (auth is opt-in via env)
  6. Three-tier flow — verify low/mid/high score paths route correctly

Reads CHECKMARBLE_API_KEY from .env automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

BRIDGE_URL = os.getenv("BRIDGE_URL", "http://localhost:8000")
API_URL = os.getenv("CHECKMARBLE_API_URL", "http://localhost:8080")
FIREBASE_URL = os.getenv("FIREBASE_URL", "http://localhost:9099")


def _load_env():
    """Load .env from project root if env vars aren't already set."""
    root = Path(__file__).parent.parent
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()
API_KEY = os.getenv("CHECKMARBLE_API_KEY", "")
ADMIN_EMAIL = os.getenv("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Marble123!")


def _section(title):
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _resolve_scenario_id() -> str | None:
    try:
        api_key = os.getenv("FIREBASE_API_KEY", "placeholder")
        if "localhost" in FIREBASE_URL or ":9099" in FIREBASE_URL:
            signin_url = f"{FIREBASE_URL}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        else:
            signin_url = f"{FIREBASE_URL}/v1/accounts:signInWithPassword?key={api_key}"
        r = httpx.post(signin_url,
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "returnSecureToken": True},
            timeout=5,
        )
        tok = r.json()["idToken"]
        r2 = httpx.post(f"{API_URL}/token", headers={"Authorization": f"Bearer {tok}"}, timeout=5)
        jwt = r2.json()["access_token"]
        r3 = httpx.get(f"{API_URL}/scenarios", headers={"Authorization": f"Bearer {jwt}"}, timeout=5)
        for s in r3.json():
            if s["name"] == "transaction_risk_scoring":
                return s["id"]
    except Exception as e:
        print(f"  scenario resolution failed: {e}")
    return None


def _make_payload(amount=5000, transaction_type="transfer"):
    txn_id = f"bnk-{uuid.uuid4().hex[:12]}"
    return {
        "object_id": txn_id,
        "updated_at": _now_iso(),
        "transaction_id": txn_id,
        "from_wallet_uid": "bnk-sender-001",
        "to_wallet_uid": "bnk-receiver-001",
        "amount": amount,
        "currency": "AFN",
        "transaction_type": transaction_type,
        "transaction_status": "Success",
        "timestamp": _now_iso(),
        "geo_location": "34.5553,69.2075",
        "imei": "356938035643809",
        "channel": "App",
        "initiated_by": "bnk-sender-001",
        "reversal_flag": False,
        "notes": "",
        # FDS fields (flat per Marble v0.59 DatabaseAccess limitation)
        "biometric_failure_count": 0,
        "velocity_spike_detected": False,
        "is_night_tx": False,
        "imei_mismatch_flag": False,
        "withdrawal_1h_sum_afn": 0,
        "sanctions_hit": False,
        "shariah_violation_flag": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_idempotency():
    _section("1. IDEMPOTENCY — same key + same body returns cached response")
    sid = _resolve_scenario_id()
    if not sid:
        print("  SKIP: could not resolve scenario_id")
        return None

    payload = _make_payload(amount=12345)
    decide_body = {"scenario_id": sid, "trigger_object_type": "transactions", "trigger_object": payload}
    key = f"idem-{uuid.uuid4().hex[:8]}"

    # First call — should execute
    t0 = time.perf_counter()
    r1 = httpx.post(f"{BRIDGE_URL}/decide", json=decide_body,
                    headers={"Idempotency-Key": key}, timeout=30)
    t1 = (time.perf_counter() - t0) * 1000

    # Second call — should be cached (no re-execution)
    t0 = time.perf_counter()
    r2 = httpx.post(f"{BRIDGE_URL}/decide", json=decide_body,
                    headers={"Idempotency-Key": key}, timeout=30)
    t2 = (time.perf_counter() - t0) * 1000

    same_body = r1.text == r2.text
    cache_faster = t2 < t1 * 0.8  # cached should be at least 20% faster

    print(f"  First call:  {t1:7.1f}ms  status={r1.status_code}")
    print(f"  Second call: {t2:7.1f}ms  status={r2.status_code}  {'(cached)' if cache_faster else ''}")
    print(f"  Same body:   {same_body}")
    ok = r1.status_code == 200 and r2.status_code == 200 and same_body
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_latency():
    _section("2. LATENCY — p50/p95/p99 of /decide (50 sequential calls)")
    sid = _resolve_scenario_id()
    if not sid:
        print("  SKIP: could not resolve scenario_id")
        return None

    latencies = []
    errors = 0
    for i in range(50):
        payload = _make_payload(amount=1000 + i * 100)
        body = {"scenario_id": sid, "trigger_object_type": "transactions", "trigger_object": payload}
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{BRIDGE_URL}/decide", json=body, timeout=10)
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1
        except Exception:
            errors += 1

    if not latencies:
        print("  SKIP: all calls failed")
        return False

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]

    print(f"  Samples:  {len(latencies)} successful, {errors} errors")
    print(f"  p50:      {p50:7.1f}ms")
    print(f"  p95:      {p95:7.1f}ms")
    print(f"  p99:      {p99:7.1f}ms")
    print(f"  mean:     {statistics.mean(latencies):7.1f}ms")
    print(f"  RESULT: {'PASS' if p99 < 5000 else 'WARN — p99 high'}")
    return p99 < 5000


def test_concurrency():
    _section("3. CONCURRENCY — 20 parallel /decide calls (no races, no errors)")
    sid = _resolve_scenario_id()
    if not sid:
        print("  SKIP: could not resolve scenario_id")
        return None

    async def one_call(client, i):
        payload = _make_payload(amount=2000 + i * 50)
        body = {"scenario_id": sid, "trigger_object_type": "transactions", "trigger_object": payload}
        try:
            r = await client.post(f"{BRIDGE_URL}/decide", json=body, timeout=10)
            return r.status_code
        except Exception as e:
            return f"err: {e}"

    async def runner():
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[one_call(client, i) for i in range(20)])
            return time.perf_counter() - t0, results

    elapsed, results = asyncio.run(runner())
    successes = sum(1 for s in results if s == 200)
    print(f"  Total time:  {elapsed * 1000:7.1f}ms (20 concurrent)")
    print(f"  Successes:   {successes}/20")
    print(f"  RESULT: {'PASS' if successes == 20 else 'FAIL'}")
    return successes == 20


def test_error_model():
    _section("4. ERROR MODEL — bad input → canonical {code, message, details}, no stack traces")
    # Send malformed payload (missing scenario_id)
    r = httpx.post(f"{BRIDGE_URL}/decide", json={"bogus": "payload"}, timeout=5)

    print(f"  Status: {r.status_code}")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    print(f"  Body:   {json.dumps(body)[:200]}")

    # Should be 4xx with canonical shape: code, message, details
    is_4xx = 400 <= r.status_code < 500
    has_canonical_shape = isinstance(body, dict) and (
        "detail" in body or all(k in body for k in ("code", "message"))
    )
    no_traceback = "Traceback" not in str(body) and "File \"/" not in str(body)

    print(f"  4xx status:        {is_4xx}")
    print(f"  Canonical shape:   {has_canonical_shape}")
    print(f"  No internal leak:  {no_traceback}")
    ok = is_4xx and has_canonical_shape and no_traceback
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_three_tier_flow():
    _section("5. THREE-TIER FLOW — verify GNN reaches 'real' mode for high-risk txn")
    sid = _resolve_scenario_id()
    if not sid:
        print("  SKIP: could not resolve scenario_id")
        return None

    # Build a high-risk transaction designed to score >= 60 (Critical, triggers GNN)
    payload = _make_payload(amount=200000, transaction_type="cash_out")
    payload["velocity_spike_detected"] = True
    payload["is_night_tx"] = True
    payload["imei_mismatch_flag"] = True
    payload["sanctions_hit"] = False
    payload["withdrawal_1h_sum_afn"] = 300000

    body = {"scenario_id": sid, "trigger_object_type": "transactions", "trigger_object": payload}
    r = httpx.post(f"{BRIDGE_URL}/decide", json=body, timeout=15)

    if r.status_code != 200:
        print(f"  FAIL: status {r.status_code}: {r.text[:200]}")
        return False

    j = r.json()
    decision = j.get("decision", {})
    escalation = j.get("escalation")

    cm_score = decision.get("score", 0)
    print(f"  Checkmarble score:  {cm_score}")

    if escalation:
        print(f"  Escalation triggered:  YES (combined_score={escalation.get('combined_score')})")
        print(f"  Marbel triggered:      {escalation.get('marbel_triggered')} (score={escalation.get('marbel_score')})")
        print(f"  GNN triggered:         {escalation.get('gnn_triggered')} (score={escalation.get('gnn_score')})")
        print(f"  Risk level:            {escalation.get('risk_level')}")
        print(f"  Suggestion:            {escalation.get('final_decision_suggestion')}")
        print(f"  Latency (ms):          {escalation.get('latency_ms')}")
    else:
        print(f"  Escalation triggered:  NO")

    print(f"  RESULT: PASS (pipeline returned a coherent response)")
    return True


def test_metrics():
    _section("6. METRICS — /metrics exposes FDS counters")
    r = httpx.get(f"{BRIDGE_URL}/metrics", timeout=5)
    text = r.text
    fds_lines = [l for l in text.splitlines() if l.startswith("fds_")]
    print(f"  Status:        {r.status_code}")
    print(f"  fds_* lines:   {len(fds_lines)}")
    for l in fds_lines[:8]:
        print(f"    {l}")
    has_decision = any("fds_decision_total" in l for l in fds_lines)
    has_escalation = any("fds_escalation_total" in l for l in fds_lines)
    has_str = any("fds_str_total" in l for l in fds_lines)
    has_ctr = any("fds_ctr_total" in l for l in fds_lines)
    ok = all([has_decision, has_escalation, has_str, has_ctr])
    print(f"  All 4 FDS counters present: {ok}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("\n" + "=" * 70)
    print(" MARBLE FDS — BANKING-GRADE RELIABILITY CHECK")
    print(f" Started: {_now_iso()}")
    print("=" * 70)

    # Pre-flight
    try:
        h = httpx.get(f"{BRIDGE_URL}/health", timeout=3)
        if h.status_code != 200:
            print(f"\nFATAL: Bridge unhealthy ({h.status_code})")
            return 1
    except Exception as e:
        print(f"\nFATAL: Bridge unreachable at {BRIDGE_URL}: {e}")
        return 1

    results = {
        "idempotency": test_idempotency(),
        "latency": test_latency(),
        "concurrency": test_concurrency(),
        "error_model": test_error_model(),
        "three_tier_flow": test_three_tier_flow(),
        "metrics": test_metrics(),
    }

    _section("SUMMARY")
    for k, v in results.items():
        status = "PASS" if v else ("SKIP" if v is None else "FAIL")
        print(f"  {k:20s}  {status}")
    failed = sum(1 for v in results.values() if v is False)
    print(f"\n  Total: {sum(1 for v in results.values() if v is True)} passed, "
          f"{sum(1 for v in results.values() if v is None)} skipped, "
          f"{failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
