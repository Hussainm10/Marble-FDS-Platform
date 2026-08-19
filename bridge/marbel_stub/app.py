"""Marbel ML Behavioral Scoring Engine — Real XGBoost V2 Service.

Loads trained XGBoost model and provides real-time fraud scoring inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Marbel Behavioral Engine", version="2.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marbel")

# --- Model Loading ---

# In Docker: /app/artifacts, in local dev: ../marbel/artifacts
ARTIFACTS_DIR = Path("/app/artifacts") if Path("/app/artifacts").exists() else Path(__file__).parent.parent / "marbel" / "artifacts"

# Global model objects (loaded at startup)
model: Optional[xgb.XGBClassifier] = None
scaler = None
calibrator = None
feature_config: dict = {}
best_threshold: float = 0.5


def load_models():
    """Load all model artifacts at startup."""
    global model, scaler, calibrator, feature_config, best_threshold

    model_path = ARTIFACTS_DIR / "xgboost_v2.json"
    scaler_path = ARTIFACTS_DIR / "scaler_v2.pkl"
    calibrator_path = ARTIFACTS_DIR / "calibrator_v2.pkl"
    config_path = ARTIFACTS_DIR / "feature_config_v2.json"

    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, falling back to stub mode")
        return False

    try:
        # Load XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        logger.info(f"Loaded XGBoost model from {model_path}")

        # Load scaler
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")

        # Load calibrator
        calibrator = joblib.load(calibrator_path)
        logger.info(f"Loaded calibrator from {calibrator_path}")

        # Load feature config
        with open(config_path) as f:
            feature_config = json.load(f)
        best_threshold = feature_config.get("best_threshold", 0.5)
        logger.info(f"Loaded config: {feature_config['feature_count']} features, threshold={best_threshold}")

        # Warmup: run one dummy inference to force XGBoost thread-pool and
        # calibrator lazy-init. Without this, the first live request takes ~70ms
        # instead of ~5ms.
        _dummy = np.zeros((1, feature_config["feature_count"]))
        _dummy_scaled = scaler.transform(_dummy)
        calibrator.predict_proba(_dummy_scaled)
        logger.info("Warmup inference complete — model ready for low-latency scoring")

        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False


# --- Request / Response models ---

class TransactionFeatures(BaseModel):
    """Raw transaction features for scoring."""
    amount: float = 0
    oldbalanceOrg: float = 0
    newbalanceOrig: float = 0
    oldbalanceDest: float = 0
    newbalanceDest: float = 0
    step: int = 1  # Time step (hour)
    type: str = "TRANSFER"  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER


class RuleInput(BaseModel):
    rule_id: str = ""
    score: float = 0
    description: str = ""


class EvaluateRequest(BaseModel):
    decision_id: str = ""
    checkmarble_score: float = 0
    trigger_object_type: str = ""
    trigger_object_id: str = ""
    rules_triggered: list[RuleInput] = Field(default_factory=list)
    # Optional: raw transaction features for direct ML scoring
    transaction: Optional[TransactionFeatures] = None


class EvaluateResponse(BaseModel):
    marbel_risk_score: float
    fraud_probability: float = 0.0
    threshold_used: float = 0.5
    is_fraud_prediction: bool = False
    contributing_factors: dict = Field(default_factory=dict)
    triggered_models: list[str] = Field(default_factory=list)
    inference_mode: str = "stub"  # "real" or "stub"


# --- Feature Engineering ---

def engineer_features(tx: TransactionFeatures) -> np.ndarray:
    """Engineer features from raw transaction data."""
    amount = tx.amount

    # Amount features
    amount_log = np.log1p(amount)
    amount_round_100 = float(amount % 100 == 0)
    amount_round_1000 = float(amount % 1000 == 0)
    amount_round_10000 = float(amount % 10000 == 0)

    # Amount percentile thresholds from PaySim training dataset
    # p50: 74,871.94, p90: 493,808.10, p99: 2,192,934.64
    AMOUNT_P50 = 74871.94
    AMOUNT_P90 = 493808.10
    AMOUNT_P99 = 2192934.64
    amount_above_median = float(amount > AMOUNT_P50)
    amount_above_p90 = float(amount > AMOUNT_P90)
    amount_above_p99 = float(amount > AMOUNT_P99)

    # Balance features (sign must match training: oldbalanceOrg - newbalanceOrig)
    balance_orig_delta = tx.oldbalanceOrg - tx.newbalanceOrig
    balance_dest_delta = tx.newbalanceDest - tx.oldbalanceDest

    # Ratio features
    amount_to_orig_balance_ratio = amount / (tx.oldbalanceOrg + 1)
    amount_to_dest_balance_ratio = amount / (tx.oldbalanceDest + 1)
    balance_orig_change_pct = balance_orig_delta / (tx.oldbalanceOrg + 1)
    balance_dest_change_pct = balance_dest_delta / (tx.oldbalanceDest + 1)

    # Suspicious patterns
    orig_zeroed_out = float(tx.newbalanceOrig == 0 and tx.oldbalanceOrg > 0)
    exact_amount_transfer = float(abs(balance_orig_delta) == amount or abs(balance_dest_delta) == amount)
    dest_new_account = float(tx.oldbalanceDest == 0 and tx.newbalanceDest > 0)
    full_balance_withdrawal = float(amount >= tx.oldbalanceOrg * 0.99 and tx.oldbalanceOrg > 0)
    dest_receives_more = float(balance_dest_delta > amount)
    orig_balance_mismatch = float(abs(tx.oldbalanceOrg - amount - tx.newbalanceOrig) > 1)

    # Time features
    hour_of_day = tx.step % 24
    day_of_week = (tx.step // 24) % 7
    day_of_month = ((tx.step // 24) % 30) + 1
    is_night = float(hour_of_day >= 22 or hour_of_day <= 5)
    is_weekend = float(day_of_week >= 5)
    is_early_morning = float(2 <= hour_of_day <= 5)

    # Transaction type one-hot encoding
    type_CASH_IN = float(tx.type == "CASH_IN")
    type_CASH_OUT = float(tx.type == "CASH_OUT")
    type_DEBIT = float(tx.type == "DEBIT")
    type_PAYMENT = float(tx.type == "PAYMENT")
    type_TRANSFER = float(tx.type == "TRANSFER")

    # Build feature vector in the same order as training
    features = np.array([
        amount,
        amount_log,
        amount_round_100,
        amount_round_1000,
        amount_round_10000,
        amount_above_median,
        amount_above_p90,
        amount_above_p99,
        tx.oldbalanceOrg,
        tx.newbalanceOrig,
        tx.oldbalanceDest,
        tx.newbalanceDest,
        balance_orig_delta,
        balance_dest_delta,
        amount_to_orig_balance_ratio,
        amount_to_dest_balance_ratio,
        balance_orig_change_pct,
        balance_dest_change_pct,
        orig_zeroed_out,
        exact_amount_transfer,
        dest_new_account,
        full_balance_withdrawal,
        dest_receives_more,
        orig_balance_mismatch,
        hour_of_day,
        day_of_week,
        day_of_month,
        is_night,
        is_weekend,
        is_early_morning,
        type_CASH_IN,
        type_CASH_OUT,
        type_DEBIT,
        type_PAYMENT,
        type_TRANSFER,
    ]).reshape(1, -1)

    return features


def get_feature_contributions(features: np.ndarray) -> dict:
    """Extract top contributing factors from features."""
    feature_names = feature_config.get("feature_names", [])
    contributions = {}

    # Identify high-risk signals
    if len(feature_names) >= 35:
        idx_map = {name: i for i, name in enumerate(feature_names)}

        if features[0, idx_map.get("orig_zeroed_out", 0)] > 0:
            contributions["account_zeroed_out"] = 0.35
        if features[0, idx_map.get("full_balance_withdrawal", 0)] > 0:
            contributions["full_balance_withdrawal"] = 0.30
        if features[0, idx_map.get("dest_new_account", 0)] > 0:
            contributions["destination_new_account"] = 0.20
        if features[0, idx_map.get("is_night", 0)] > 0:
            contributions["night_transaction"] = 0.15
        if features[0, idx_map.get("orig_balance_mismatch", 0)] > 0:
            contributions["balance_mismatch"] = 0.25
        if features[0, idx_map.get("amount_to_orig_balance_ratio", 0)] > 0.9:
            contributions["high_amount_ratio"] = 0.28

    return contributions


# --- Stub Scoring (fallback) ---

HIGH_RISK_PATTERNS = {
    "velocity": 0.27, "vpn": 0.18, "blacklist": 0.22, "sanctions": 0.25,
    "reversal": 0.15, "mismatch": 0.12, "biometric": 0.14, "dormant": 0.10,
    "float": 0.08, "shariah": 0.06,
}


def compute_stub_score(req: EvaluateRequest) -> EvaluateResponse:
    """Fallback stub scoring when models not loaded."""
    rules = req.rules_triggered
    base = req.checkmarble_score * 1.1
    rule_boost = len(rules) * 3
    raw_score = base + rule_boost
    marbel_score = min(100.0, round(raw_score, 1))

    contributions = {}
    for rule in rules:
        combined = f"{rule.rule_id} {rule.description}".lower()
        for pattern, weight in HIGH_RISK_PATTERNS.items():
            if pattern in combined and pattern not in contributions:
                contributions[pattern] = weight

    models = ["behavioral_profiling_model_v2"]
    if any("velocity" in f or "spike" in f for f in contributions):
        models.append("velocity_anomaly_detector")

    return EvaluateResponse(
        marbel_risk_score=marbel_score,
        fraud_probability=marbel_score / 100.0,
        threshold_used=0.5,
        is_fraud_prediction=marbel_score > 50,
        contributing_factors=contributions,
        triggered_models=models,
        inference_mode="stub",
    )


def compute_real_score(req: EvaluateRequest) -> EvaluateResponse:
    """Real ML scoring using trained XGBoost model."""
    if req.transaction is None:
        # No raw transaction data; use stub with adjusted score
        stub_result = compute_stub_score(req)
        stub_result.inference_mode = "stub_no_features"
        return stub_result

    # Engineer features
    features = engineer_features(req.transaction)

    # Scale features
    features_scaled = scaler.transform(features)

    # Get calibrated probability (calibrator wraps model and expects features)
    calibrated_proba = calibrator.predict_proba(features_scaled)[0, 1]

    # Convert to 0-100 risk score (ensure Python float, not numpy)
    risk_score = float(round(calibrated_proba * 100, 1))

    # Determine fraud prediction (ensure Python bool, not numpy)
    is_fraud = bool(calibrated_proba >= best_threshold)

    # Get contributing factors
    contributions = get_feature_contributions(features)

    return EvaluateResponse(
        marbel_risk_score=risk_score,
        fraud_probability=float(round(calibrated_proba, 4)),
        threshold_used=float(best_threshold),
        is_fraud_prediction=is_fraud,
        contributing_factors=contributions,
        triggered_models=["xgboost_behavioral_v2", "isotonic_calibrator"],
        inference_mode="real",
    )


# --- Endpoints ---

@app.on_event("startup")
async def startup():
    """Load models on startup."""
    success = load_models()
    if success:
        logger.info("Marbel service started with REAL XGBoost V2 model")
    else:
        logger.warning("Marbel service started in STUB mode (models not loaded)")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    logger.info(
        "Marbel evaluate: decision=%s checkmarble_score=%s rules=%d has_transaction=%s",
        req.decision_id,
        req.checkmarble_score,
        len(req.rules_triggered),
        req.transaction is not None,
    )

    if model is not None and req.transaction is not None:
        result = compute_real_score(req)
    else:
        result = compute_stub_score(req)

    logger.info("Marbel result: score=%s prob=%s mode=%s",
                result.marbel_risk_score, result.fraud_probability, result.inference_mode)
    return result


@app.post("/score_transaction")
async def score_transaction(tx: TransactionFeatures):
    """Direct transaction scoring endpoint (bypasses Checkmarble integration)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = engineer_features(tx)
    features_scaled = scaler.transform(features)
    # Use calibrated classifier (expects features, not probabilities)
    calibrated_proba = calibrator.predict_proba(features_scaled)[0, 1]

    return {
        "fraud_probability": float(round(calibrated_proba, 4)),
        "risk_score": float(round(calibrated_proba * 100, 1)),
        "is_fraud": bool(calibrated_proba >= best_threshold),
        "threshold": float(best_threshold),
        "contributing_factors": get_feature_contributions(features),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "marbel",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "inference_mode": "real" if model is not None else "stub",
    }


@app.get("/model_info")
async def model_info():
    """Return model metadata and metrics."""
    if not feature_config:
        return {"error": "Model not loaded"}

    return {
        "version": feature_config.get("version"),
        "trained_at": feature_config.get("trained_at"),
        "feature_count": feature_config.get("feature_count"),
        "metrics": feature_config.get("metrics"),
        "best_threshold": best_threshold,
    }
