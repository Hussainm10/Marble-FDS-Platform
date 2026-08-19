"""Integration tests for real ML model inference.

Tests verify that Marbel (XGBoost v2) and GNN (HybridGNN v4) services
are running in real inference mode with expected performance.
"""

import time
from typing import Any

import pytest
import requests

# Service endpoints
MARBEL_URL = "http://localhost:5000"
GNN_URL = "http://localhost:5001"

# Timeouts
REQUEST_TIMEOUT = 5.0
MARBEL_LATENCY_P99_MS = 50
GNN_LATENCY_P99_MS = 100


def is_service_running(url: str) -> bool:
    """Check if a service is running and healthy."""
    try:
        resp = requests.get(f"{url}/health", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def marbel_available():
    """Skip tests if Marbel service is not running."""
    if not is_service_running(MARBEL_URL):
        pytest.skip("Marbel service not running on localhost:5000")


@pytest.fixture(scope="module")
def gnn_available():
    """Skip tests if GNN service is not running."""
    if not is_service_running(GNN_URL):
        pytest.skip("GNN service not running on localhost:5001")


# =============================================================================
# MARBEL (XGBoost V2) TESTS
# =============================================================================

class TestMarbelHealth:
    """Health and configuration tests for Marbel service."""

    def test_health_endpoint(self, marbel_available):
        """Test /health endpoint returns expected fields."""
        resp = requests.get(f"{MARBEL_URL}/health", timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "ok"
        assert data["service"] == "marbel"
        assert data["version"] == "2.0.0"
        assert "model_loaded" in data
        assert "inference_mode" in data

    def test_inference_mode_real(self, marbel_available):
        """Verify Marbel is running in real inference mode."""
        resp = requests.get(f"{MARBEL_URL}/health", timeout=REQUEST_TIMEOUT)
        data = resp.json()

        assert data["model_loaded"] is True, "Model should be loaded"
        assert data["inference_mode"] == "real", "Should be in real inference mode"

    def test_model_info(self, marbel_available):
        """Verify model info endpoint returns expected metadata."""
        resp = requests.get(f"{MARBEL_URL}/model_info", timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()

        assert data.get("version") == "2.0.0"
        assert data.get("feature_count") == 35
        assert "metrics" in data
        assert data["metrics"]["auc_roc"] > 0.99, "AUC-ROC should be > 0.99"


class TestMarbelFraudDetection:
    """Fraud detection accuracy tests for Marbel."""

    def test_high_risk_cash_out(self, marbel_available):
        """High-risk CASH_OUT should score high (account zeroed out)."""
        payload = {
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 500000,
            "step": 3,
            "type": "CASH_OUT",
        }

        resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["fraud_probability"] > 0.5, "High-risk transaction should have prob > 0.5"
        assert data["is_fraud"] is True, "Should be flagged as fraud"

    def test_high_risk_transfer(self, marbel_available):
        """High-risk TRANSFER with full balance withdrawal."""
        payload = {
            "amount": 1000000,
            "oldbalanceOrg": 1000000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 1000000,
            "step": 3,
            "type": "TRANSFER",
        }

        resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["fraud_probability"] > 0.5

    def test_low_risk_payment(self, marbel_available):
        """Normal payment should score low."""
        payload = {
            "amount": 50,
            "oldbalanceOrg": 10000,
            "newbalanceOrig": 9950,
            "oldbalanceDest": 5000,
            "newbalanceDest": 5050,
            "step": 12,
            "type": "PAYMENT",
        }

        resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["fraud_probability"] < 0.3, "Low-risk payment should have low probability"
        assert data["is_fraud"] is False, "Should not be flagged as fraud"

    def test_low_risk_cash_in(self, marbel_available):
        """Normal CASH_IN should score low."""
        payload = {
            "amount": 1000,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 6000,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
            "step": 10,
            "type": "CASH_IN",
        }

        resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["fraud_probability"] < 0.5

    def test_amount_above_p99_threshold(self, marbel_available):
        """Very large amount (above p99) should influence scoring."""
        payload = {
            "amount": 3000000,  # Above p99 threshold (2,192,934)
            "oldbalanceOrg": 5000000,
            "newbalanceOrig": 2000000,
            "oldbalanceDest": 1000000,
            "newbalanceDest": 4000000,
            "step": 5,
            "type": "TRANSFER",
        }

        resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        # Just verify it processes correctly
        assert "fraud_probability" in resp.json()


class TestMarbelLatency:
    """Latency benchmark tests for Marbel."""

    def test_latency_p99(self, marbel_available):
        """Verify p99 latency is under 50ms."""
        payload = {
            "amount": 100000,
            "oldbalanceOrg": 200000,
            "newbalanceOrig": 100000,
            "oldbalanceDest": 50000,
            "newbalanceDest": 150000,
            "step": 5,
            "type": "TRANSFER",
        }

        # Warmup: discard first 10 requests so XGBoost thread-pool and calibrator
        # lazy-init don't pollute the benchmark (cold-start spike is ~70ms,
        # warm steady-state is ~5ms).
        for _ in range(10):
            requests.post(f"{MARBEL_URL}/score_transaction", json=payload, timeout=REQUEST_TIMEOUT)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            resp = requests.post(
                f"{MARBEL_URL}/score_transaction",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert resp.status_code == 200

        latencies.sort()
        p99 = latencies[98]  # 99th percentile of 100 samples

        assert p99 < MARBEL_LATENCY_P99_MS, f"p99 latency {p99:.1f}ms exceeds {MARBEL_LATENCY_P99_MS}ms"


class TestMarbelEvaluateEndpoint:
    """Tests for /evaluate endpoint (Checkmarble integration)."""

    def test_evaluate_with_transaction(self, marbel_available):
        """Evaluate endpoint with transaction features."""
        payload = {
            "decision_id": "test-123",
            "checkmarble_score": 30,
            "trigger_object_type": "Transaction",
            "trigger_object_id": "tx-456",
            "rules_triggered": [
                {"rule_id": "r1", "score": 10, "description": "velocity check"}
            ],
            "transaction": {
                "amount": 500000,
                "oldbalanceOrg": 500000,
                "newbalanceOrig": 0,
                "oldbalanceDest": 0,
                "newbalanceDest": 500000,
                "step": 3,
                "type": "CASH_OUT",
            },
        }

        resp = requests.post(
            f"{MARBEL_URL}/evaluate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["inference_mode"] == "real"
        assert "fraud_probability" in data
        assert "marbel_risk_score" in data

    def test_evaluate_without_transaction_fallback(self, marbel_available):
        """Evaluate without transaction features falls back to stub scoring."""
        payload = {
            "decision_id": "test-789",
            "checkmarble_score": 50,
            "trigger_object_type": "Transaction",
            "trigger_object_id": "tx-999",
            "rules_triggered": [],
        }

        resp = requests.post(
            f"{MARBEL_URL}/evaluate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["inference_mode"] in ["stub", "stub_no_features"]


# =============================================================================
# GNN (HybridGNN V4) TESTS
# =============================================================================

class TestGNNHealth:
    """Health and configuration tests for GNN service."""

    def test_health_endpoint(self, gnn_available):
        """Test /health endpoint returns expected fields."""
        resp = requests.get(f"{GNN_URL}/health", timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "ok"
        assert data["service"] == "gnn"
        assert data["version"] == "4.0.0"
        assert "model_loaded" in data
        assert "inference_mode" in data

    def test_inference_mode_real(self, gnn_available):
        """Verify GNN is running in real inference mode."""
        resp = requests.get(f"{GNN_URL}/health", timeout=REQUEST_TIMEOUT)
        data = resp.json()

        assert data["model_loaded"] is True, "Model should be loaded"
        assert data["inference_mode"] == "real", "Should be in real inference mode"
        assert data.get("model_type") == "HybridGNN", "Should be HybridGNN model"

    def test_model_info(self, gnn_available):
        """Verify model info endpoint returns expected metadata."""
        resp = requests.get(f"{GNN_URL}/model_info", timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()

        assert data.get("version") == "4.0.0"
        assert data.get("architecture", {}).get("in_channels") == 42
        assert data.get("best_threshold") == 0.4


class TestGNNScoring:
    """Scoring tests for GNN service."""

    def _make_features(self, base_value: float = 0.0) -> list[float]:
        """Generate a 42-dimensional feature vector matching v4's PaySim contract."""
        return [base_value] * 42

    def test_score_node_valid_features(self, gnn_available):
        """Test /score_node with valid 42 features."""
        features = self._make_features(0.1)

        resp = requests.post(
            f"{GNN_URL}/score_node",
            json={"features": features, "neighbors": []},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "fraud_probability" in data
        assert "risk_score" in data
        assert "is_fraud" in data
        assert 0 <= data["fraud_probability"] <= 1

    def test_score_node_invalid_features(self, gnn_available):
        """Test /score_node rejects wrong feature count."""
        features = [0.1] * 41  # Wrong: should be 42

        resp = requests.post(
            f"{GNN_URL}/score_node",
            json={"features": features, "neighbors": []},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400

    def test_score_node_with_neighbors(self, gnn_available):
        """Test scoring with neighbor features."""
        features = self._make_features(0.2)
        neighbors = [self._make_features(0.1), self._make_features(0.15)]

        resp = requests.post(
            f"{GNN_URL}/score_node",
            json={"features": features, "neighbors": neighbors},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "fraud_probability" in data


class TestGNNLatency:
    """Latency benchmark tests for GNN."""

    def test_latency_p99(self, gnn_available):
        """Verify p99 latency is under 100ms."""
        features = [0.1] * 42

        latencies = []
        for _ in range(50):  # Fewer iterations for heavier model
            start = time.perf_counter()
            resp = requests.post(
                f"{GNN_URL}/score_node",
                json={"features": features, "neighbors": []},
                timeout=REQUEST_TIMEOUT,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert resp.status_code == 200

        latencies.sort()
        p99 = latencies[48]  # 99th percentile of 50 samples

        assert p99 < GNN_LATENCY_P99_MS, f"p99 latency {p99:.1f}ms exceeds {GNN_LATENCY_P99_MS}ms"


class TestGNNEvaluateEndpoint:
    """Tests for /evaluate endpoint."""

    def _make_features(self) -> list[float]:
        return [0.1] * 42

    def test_evaluate_with_node_data(self, gnn_available):
        """Evaluate endpoint with node features."""
        payload = {
            "decision_id": "test-gnn-123",
            "checkmarble_score": 40,
            "marbel_score": 60,
            "trigger_object_type": "Transaction",
            "trigger_object_id": "tx-gnn-456",
            "entity_id": "entity-789",
            "node_data": {
                "features": self._make_features(),
                "neighbors": [],
            },
        }

        resp = requests.post(
            f"{GNN_URL}/evaluate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["inference_mode"] == "real"
        assert "fraud_probability" in data
        assert "enhanced_score" in data

    def test_evaluate_without_node_data_fallback(self, gnn_available):
        """Evaluate without node data falls back to stub scoring."""
        payload = {
            "decision_id": "test-gnn-789",
            "checkmarble_score": 50,
            "marbel_score": 70,
            "trigger_object_type": "Transaction",
            "trigger_object_id": "tx-gnn-999",
            "entity_id": "entity-000",
        }

        resp = requests.post(
            f"{GNN_URL}/evaluate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["inference_mode"] in ["stub", "stub_no_features"]


# =============================================================================
# COMBINED TESTS
# =============================================================================

class TestBothServicesRunning:
    """Tests requiring both services."""

    def test_both_services_healthy(self, marbel_available, gnn_available):
        """Verify both services are healthy and in real mode."""
        marbel_health = requests.get(f"{MARBEL_URL}/health", timeout=REQUEST_TIMEOUT).json()
        gnn_health = requests.get(f"{GNN_URL}/health", timeout=REQUEST_TIMEOUT).json()

        assert marbel_health["inference_mode"] == "real"
        assert gnn_health["inference_mode"] == "real"

    def test_pipeline_scoring(self, marbel_available, gnn_available):
        """Test a simulated pipeline: Marbel -> GNN scoring."""
        # First: Score with Marbel
        marbel_payload = {
            "amount": 250000,
            "oldbalanceOrg": 300000,
            "newbalanceOrig": 50000,
            "oldbalanceDest": 10000,
            "newbalanceDest": 260000,
            "step": 8,
            "type": "TRANSFER",
        }

        marbel_resp = requests.post(
            f"{MARBEL_URL}/score_transaction",
            json=marbel_payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert marbel_resp.status_code == 200
        marbel_score = marbel_resp.json()["risk_score"]

        # Second: Score with GNN (using stub since we don't have real graph features)
        gnn_payload = {
            "decision_id": "pipeline-test",
            "checkmarble_score": 30,
            "marbel_score": marbel_score,
            "entity_id": "test-entity",
        }

        gnn_resp = requests.post(
            f"{GNN_URL}/evaluate",
            json=gnn_payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert gnn_resp.status_code == 200
        assert "enhanced_score" in gnn_resp.json()
