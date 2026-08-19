"""Complete the Marbel training - run calibration, evaluation, and save artifacts.

This script picks up where train_marbel.py failed (at calibration step).
Uses the best hyperparameters found during tuning.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PAYSIM_FILE = DATA_DIR / "paysim" / "PS_20174392719_1491204439457_log.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "marbel" / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Best hyperparameters from the completed tuning
BEST_PARAMS = {
    "n_estimators": 149,
    "max_depth": 8,
    "learning_rate": 0.01005403021778705,
    "min_child_weight": 3,
    "subsample": 0.8873900957215456,
    "colsample_bytree": 0.9969301382638496,
    "gamma": 4.762056072749309,
    "reg_alpha": 0.031501351500099424,
    "reg_lambda": 0.31050673796449324,
}


def load_and_preprocess_data():
    """Load PaySim data and engineer features."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)

    df = pd.read_csv(PAYSIM_FILE)
    print(f"Loaded {len(df):,} transactions")

    # Transaction type encoding
    type_dummies = pd.get_dummies(df["type"], prefix="type")

    # Amount features
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_round_100"] = (df["amount"] % 100 == 0).astype(int)
    df["amount_round_1000"] = (df["amount"] % 1000 == 0).astype(int)

    # Balance features
    df["balance_orig_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balance_orig_change_ratio"] = df["balance_orig_delta"] / (df["oldbalanceOrg"] + 1)

    # Suspicious patterns
    df["orig_zeroed_out"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["exact_amount_transfer"] = (df["balance_orig_delta"] == df["amount"]).astype(int)
    df["dest_new_account"] = (df["oldbalanceDest"] == 0).astype(int)

    # Time features
    df["hour_of_day"] = df["step"] % 24
    df["day_of_month"] = (df["step"] // 24) % 30
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_month"] % 7 >= 5).astype(int)

    feature_cols = [
        "amount", "amount_log", "amount_round_100", "amount_round_1000",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "balance_orig_delta", "balance_dest_delta", "amount_to_balance_ratio",
        "balance_orig_change_ratio", "orig_zeroed_out", "exact_amount_transfer",
        "dest_new_account", "hour_of_day", "day_of_month", "is_night", "is_weekend",
        "isFlaggedFraud",
    ]

    X = pd.concat([df[feature_cols], type_dummies], axis=1)
    y = df["isFraud"]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_names = list(X.columns)
    print(f"Total features: {len(feature_names)}")

    return X.values, y.values, feature_names


def split_data(X, y):
    """Split into train/val/test."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Training: {len(X_train):,}, Validation: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE."""
    from imblearn.over_sampling import SMOTE
    fraud_count = sum(y_train == 1)
    normal_count = sum(y_train == 0)
    target_fraud = int(normal_count * 0.1)
    if target_fraud > fraud_count:
        smote = SMOTE(sampling_strategy={1: target_fraud}, random_state=RANDOM_STATE, k_neighbors=5)
        return smote.fit_resample(X_train, y_train)
    return X_train, y_train


def train_model(X_train, y_train, X_val, y_val):
    """Train with best hyperparameters."""
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("STEP 2: Training Final Model with Best Hyperparameters")
    print("=" * 60)
    print(f"Best params: {BEST_PARAMS}")

    params = {
        **BEST_PARAMS,
        "scale_pos_weight": sum(y_train == 0) / max(sum(y_train == 1), 1),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    print(f"Training on {len(X_combined):,} samples...")
    model.fit(X_combined, y_combined, verbose=False)
    print("Training complete!")
    return model


def calibrate_model(model, X_val, y_val):
    """Calibrate probabilities - fixed version."""
    print("\n" + "=" * 60)
    print("STEP 3: Calibrating Probabilities")
    print("=" * 60)

    # Use cv=5 instead of 'prefit' for newer sklearn versions
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrator.fit(X_val, y_val)

    raw_proba = model.predict_proba(X_val)[:, 1]
    cal_proba = calibrator.predict_proba(X_val)[:, 1]
    print(f"Raw probabilities - mean: {raw_proba.mean():.4f}, std: {raw_proba.std():.4f}")
    print(f"Calibrated probs  - mean: {cal_proba.mean():.4f}, std: {cal_proba.std():.4f}")
    return calibrator


def evaluate_model(model, calibrator, X_test, y_test, feature_names):
    """Evaluate on test set."""
    print("\n" + "=" * 60)
    print("STEP 4: RIGOROUS EVALUATION ON TEST SET")
    print("=" * 60)

    y_pred_proba = calibrator.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n--- Metrics at Different Thresholds ---")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)

    best_f1, best_threshold = 0, 0.5
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    print(f"\n--- Core Metrics ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Best F1: {best_f1:.4f} (threshold {best_threshold})")

    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\n--- Confusion Matrix ---")
    print(f"                 Predicted Normal  Predicted Fraud")
    print(f"Actual Normal    {cm[0,0]:>14}  {cm[0,1]:>14}")
    print(f"Actual Fraud     {cm[1,0]:>14}  {cm[1,1]:>14}")

    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_best, target_names=["Normal", "Fraud"]))

    # Feature importance
    print("\n--- Top 15 Important Features ---")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:<30} {importance[idx]:.4f}")

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "precision_at_best": float(precision_score(y_test, y_pred_best, zero_division=0)),
        "recall_at_best": float(recall_score(y_test, y_pred_best, zero_division=0)),
        "test_samples": int(len(y_test)),
        "test_fraud_count": int(sum(y_test)),
    }


def save_artifacts(model, calibrator, scaler, feature_names, metrics):
    """Save all artifacts."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving Model Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model
    model_path = ARTIFACTS_DIR / "xgboost_v1.json"
    model.save_model(str(model_path))
    print(f"Saved: {model_path}")

    # Save calibrator
    calibrator_path = ARTIFACTS_DIR / "calibrator.pkl"
    joblib.dump(calibrator, calibrator_path)
    print(f"Saved: {calibrator_path}")

    # Save scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")

    # Save config
    config = {
        "version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "paysim",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "hyperparameters": BEST_PARAMS,
        "metrics": metrics,
        "best_threshold": metrics["best_threshold"],
    }
    config_path = ARTIFACTS_DIR / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # Save SHAP explainer
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, ARTIFACTS_DIR / "shap_explainer.pkl")
        print(f"Saved: shap_explainer.pkl")
    except Exception as e:
        print(f"SHAP explainer skipped: {e}")

    print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")


def main():
    print("=" * 60)
    print("COMPLETING MARBEL TRAINING")
    print("=" * 60)

    # Load data
    X, y, feature_names = load_and_preprocess_data()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

    # SMOTE
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal):,} samples")

    # Train
    model = train_model(X_train_bal, y_train_bal, X_val, y_val)

    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, calibrator, X_test, y_test, feature_names)

    # Save
    save_artifacts(model, calibrator, scaler, feature_names, metrics)

    print("\n" + "=" * 60)
    print("MARBEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Final AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"Final F1:      {metrics['best_f1']:.4f}")


if __name__ == "__main__":
    main()
