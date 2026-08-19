"""End-to-end integration test: ingest -> decide -> verify outcome.

Run: pytest tests/test_integration.py -v
Requires: All services running (docker compose up -d)
"""

import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest

API_URL = os.getenv("CHECKMARBLE_API_URL", "http://localhost:8080")
BRIDGE_URL = os.getenv("BRIDGE_URL", "http://localhost:8000")
API_KEY = os.getenv("CHECKMARBLE_API_KEY", "")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

_scenario_ids: dict[str, str] = {}


def _uid():
    return f"e2e-{uuid.uuid4().hex[:12]}"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _get_jwt():
    firebase_url = os.getenv("FIREBASE_URL", "http://localhost:9099")
    api_key = os.getenv("FIREBASE_API_KEY", "placeholder")
    admin_email = os.getenv("CREATE_ORG_ADMIN_EMAIL", "admin@example.com")
    admin_password = os.environ["ADMIN_PASSWORD"]  # must be set; no default
    # Emulator quirk: serves REST API at /identitytoolkit.googleapis.com/...; real Firebase serves at /v1/...
    if "localhost" in firebase_url or ":9099" in firebase_url:
        signin_url = f"{firebase_url}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    else:
        signin_url = f"{firebase_url}/v1/accounts:signInWithPassword?key={api_key}"
    r = httpx.post(signin_url,
        json={"email": admin_email, "password": admin_password, "returnSecureToken": True},
    )
    id_token = r.json()["idToken"]
    r = httpx.post(f"{API_URL}/token", headers={"Authorization": f"Bearer {id_token}"})
    return r.json()["access_token"]


@pytest.fixture(autouse=True, scope="session")
def load_scenario_ids():
    try:
        jwt = _get_jwt()
        r = httpx.get(f"{API_URL}/scenarios", headers={"Authorization": f"Bearer {jwt}"}, timeout=10)
        if r.status_code == 200:
            for s in r.json():
                _scenario_ids[s["name"]] = s["id"]
    except Exception:
        pytest.skip("Cannot connect to Marble API")


@pytest.fixture(autouse=True)
def check_services():
    """Skip if services are not available."""
    for name, url in [("Marble", f"{API_URL}/liveness"), ("Bridge", f"{BRIDGE_URL}/health")]:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code != 200:
                pytest.skip(f"{name} not available")
        except httpx.ConnectError:
            pytest.skip(f"{name} not reachable")


class TestEndToEnd:
    """Full pipeline: ingest → decide → verify outcome."""

    def test_safe_transaction_flow(self):
        """A clean transaction with no risk flags should approve."""
        txn_id = _uid()
        scenario_id = _scenario_ids.get("transaction_risk_scoring")
        if not scenario_id:
            pytest.skip("transaction_risk_scoring scenario not found")

        # Ingest transaction via API
        payload = {
            "object_id": txn_id,
            "updated_at": _now(),
            "transaction_id": txn_id,
            "from_wallet_uid": "safe-user-001",
            "to_wallet_uid": "merchant-safe-001",
            "amount": 5000,
            "currency": "AFN",
            "transaction_type": "transfer",
            "transaction_status": "Success",
            "timestamp": "2026-01-30T14:00:00Z",
            "geo_location": "34.5553,69.2075",
            "imei": "356938035643809",
            "channel": "App",
            "initiated_by": "safe-user-001",
            "reversal_flag": False,
            "notes": "",
        }
        r = httpx.post(f"{API_URL}/ingestion/transactions", json=payload, headers=HEADERS, timeout=30)
        assert r.status_code == 201

        # Request decision with full trigger_object
        decision_payload = {
            "scenario_id": scenario_id,
            "object_type": "transactions",
            "trigger_object": payload,
        }
        r = httpx.post(f"{API_URL}/decisions", json=decision_payload, headers=HEADERS, timeout=30)
        assert r.status_code == 200

        result = r.json()
        assert result["score"] <= 24, f"Clean txn should approve, got score {result['score']}"
        assert result["outcome"] == "approve"

    def test_risky_transaction_via_bridge(self):
        """A risky transaction through the bridge should trigger escalation."""
        scenario_id = _scenario_ids.get("transaction_risk_scoring")
        if not scenario_id:
            pytest.skip("transaction_risk_scoring scenario not found")

        txn_id = _uid()
        # Risky transaction trigger_object. FDS pre-computed features are
        # flattened into the trigger (the operator backend is responsible
        # for merging fds_input_features → transaction payload at /decide time).
        trigger = {
            "object_id": txn_id,
            "updated_at": _now(),
            "transaction_id": txn_id,
            "from_wallet_uid": "risky-user-001",
            "to_wallet_uid": "mule-001",
            "amount": 150000,
            "currency": "AFN",
            "transaction_type": "withdrawal",
            "transaction_status": "Pending",
            "timestamp": "2026-01-30T02:00:00Z",
            "geo_location": "",
            "imei": "999888777666555",
            "channel": "POS",
            "reversal_flag": True,
            "reversal_reason": "Disputed",
            "notes": "night blacklisted",
            # FDS pre-computed features (should push score well above 40):
            "velocity_spike_detected": True,   # RSK_VEL_060 +15
            "is_night_tx": True,               # RSK_TME_062 +8
            "imei_mismatch_flag": True,        # RSK_DEV_063 +20
            "withdrawal_1h_sum_afn": 150000,   # contributes to CMP_STR_020
        }

        r = httpx.post(
            f"{BRIDGE_URL}/decide",
            json={
                "scenario_id": scenario_id,
                "trigger_object_type": "transactions",
                "trigger_object": trigger,
            },
            timeout=30,
        )
        assert r.status_code == 200
        result = r.json()

        decision = result.get("decision", {})
        score = decision.get("score", 0)
        assert score >= 40, f"Risky txn should score >= 40, got {score}"

        # Escalation should have been triggered
        assert "escalation" in result, "Escalation should be triggered for score >= 40"

    def test_webhook_processing(self):
        """Webhook endpoint should accept and process events."""
        r = httpx.post(
            f"{BRIDGE_URL}/webhooks/checkmarble",
            json={
                "event": "decision.created",
                "decision_id": "test-dec-001",
                "outcome": "review",
                "score": 30,
                "reviewed_by": "",
                "notes": "",
            },
            timeout=30,
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") in ("received", "acknowledged", "escalated", "escalation_failed")

    def test_bridge_health(self):
        """Bridge health endpoint should report status."""
        r = httpx.get(f"{BRIDGE_URL}/health", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["checkmarble_connected"] is True

    def test_ingest_via_bridge(self):
        """Bridge ingest proxy should forward to Marble."""
        uid = _uid()
        r = httpx.post(
            f"{BRIDGE_URL}/ingest/individual_users",
            json={
                "object_id": uid,
                "data": {
                    "user_id": uid,
                    "full_name": "Bridge Ingest Test",
                    "kyc_level": "L1",
                    "account_status": "Active",
                },
            },
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json().get("status") == "ingested"
