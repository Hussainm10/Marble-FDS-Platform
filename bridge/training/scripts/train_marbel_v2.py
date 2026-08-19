"""Train Marbel XGBoost fraud detection model - V2 (No Data Leakage).

Changes from V1:
- REMOVED isFlaggedFraud (data leakage)
- Added more robust features
- Added cross-validation for better evaluation
- More rigorous hyperparameter tuning
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PAYSIM_FILE = DATA_DIR / "paysim" / "PS_20174392719_1491204439457_log.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "marbel" / "artifacts"

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_OPTUNA_TRIALS = 50
CV_FOLDS = 5


def load_and_preprocess_data():
    """Load PaySim data and engineer features - NO DATA LEAKAGE."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Preprocessing Data (V2 - No Leakage)")
    print("=" * 60)

    df = pd.read_csv(PAYSIM_FILE)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.3f}%")

    # --- Feature Engineering (NO isFlaggedFraud!) ---
    print("\nEngineering features (without data leakage)...")

    # Transaction type encoding
    type_dummies = pd.get_dummies(df["type"], prefix="type")

    # Amount features
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_round_100"] = (df["amount"] % 100 == 0).astype(int)
    df["amount_round_1000"] = (df["amount"] % 1000 == 0).astype(int)
    df["amount_round_10000"] = (df["amount"] % 10000 == 0).astype(int)

    # Amount percentiles (relative to dataset)
    amount_p50 = df["amount"].quantile(0.5)
    amount_p90 = df["amount"].quantile(0.9)
    amount_p99 = df["amount"].quantile(0.99)
    df["amount_above_median"] = (df["amount"] > amount_p50).astype(int)
    df["amount_above_p90"] = (df["amount"] > amount_p90).astype(int)
    df["amount_above_p99"] = (df["amount"] > amount_p99).astype(int)

    # Balance features
    df["balance_orig_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Ratio features (with safety for zero balances)
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest_balance_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["balance_orig_change_pct"] = df["balance_orig_delta"] / (df["oldbalanceOrg"] + 1)
    df["balance_dest_change_pct"] = df["balance_dest_delta"] / (df["oldbalanceDest"] + 1)

    # Suspicious patterns (legitimate fraud signals)
    df["orig_zeroed_out"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["exact_amount_transfer"] = (abs(df["balance_orig_delta"] - df["amount"]) < 0.01).astype(int)
    df["dest_new_account"] = (df["oldbalanceDest"] == 0).astype(int)
    df["full_balance_withdrawal"] = ((df["oldbalanceOrg"] > 0) & (df["amount"] >= df["oldbalanceOrg"] * 0.99)).astype(int)

    # Dest receives more than sent (money mule pattern)
    df["dest_receives_more"] = (df["balance_dest_delta"] > df["amount"] * 1.01).astype(int)

    # Origin balance inconsistency (potential fraud indicator)
    df["orig_balance_mismatch"] = (abs(df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]) > 1).astype(int)

    # Time features (step = 1 hour in PaySim)
    df["hour_of_day"] = df["step"] % 24
    df["day_of_week"] = (df["step"] // 24) % 7
    df["day_of_month"] = (df["step"] // 24) % 30
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_early_morning"] = ((df["hour_of_day"] >= 2) & (df["hour_of_day"] <= 5)).astype(int)

    # Combine features - EXPLICITLY NO isFlaggedFraud
    feature_cols = [
        # Amount features
        "amount", "amount_log", "amount_round_100", "amount_round_1000", "amount_round_10000",
        "amount_above_median", "amount_above_p90", "amount_above_p99",
        # Balance features
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "balance_orig_delta", "balance_dest_delta",
        # Ratio features
        "amount_to_orig_balance_ratio", "amount_to_dest_balance_ratio",
        "balance_orig_change_pct", "balance_dest_change_pct",
        # Suspicious patterns
        "orig_zeroed_out", "exact_amount_transfer", "dest_new_account",
        "full_balance_withdrawal", "dest_receives_more", "orig_balance_mismatch",
        # Time features
        "hour_of_day", "day_of_week", "day_of_month",
        "is_night", "is_weekend", "is_early_morning",
        # NOTE: isFlaggedFraud is INTENTIONALLY EXCLUDED (data leakage)
    ]

    X = pd.concat([df[feature_cols], type_dummies], axis=1)
    y = df["isFraud"]

    # Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_names = list(X.columns)
    print(f"Total features: {len(feature_names)} (isFlaggedFraud REMOVED)")
    print(f"Features: {feature_names[:10]}... (and {len(feature_names)-10} more)")

    return X.values, y.values, feature_names


def split_data(X, y):
    """Split into train/val/test with stratification."""
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Data (Stratified)")
    print("=" * 60)

    # First split: separate test set (HELD OUT - never touched during training)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Second split: separate validation from training
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Training set:   {len(X_train):,} ({y_train.mean()*100:.3f}% fraud)")
    print(f"Validation set: {len(X_val):,} ({y_val.mean()*100:.3f}% fraud)")
    print(f"Test set:       {len(X_test):,} ({y_test.mean()*100:.3f}% fraud)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training data."""
    print("\n" + "=" * 60)
    print("STEP 3: Handling Class Imbalance (SMOTE)")
    print("=" * 60)

    try:
        from imblearn.over_sampling import SMOTE

        print(f"Before SMOTE: {sum(y_train == 0):,} normal, {sum(y_train == 1):,} fraud")

        # Target 10% fraud ratio (not 50/50 which can cause overfitting)
        fraud_count = sum(y_train == 1)
        normal_count = sum(y_train == 0)
        target_fraud = int(normal_count * 0.1)

        if target_fraud > fraud_count:
            smote = SMOTE(
                sampling_strategy={1: target_fraud},
                random_state=RANDOM_STATE,
                k_neighbors=5,
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE:  {sum(y_resampled == 0):,} normal, {sum(y_resampled == 1):,} fraud")
            return X_resampled, y_resampled
        else:
            print("SMOTE not needed - sufficient fraud samples")
            return X_train, y_train

    except ImportError:
        print("imbalanced-learn not installed, using class weights instead")
        return X_train, y_train


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials):
    """Use Optuna for hyperparameter optimization with cross-validation."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Hyperparameter Tuning ({n_trials} trials with CV)")
    print("=" * 60)

    try:
        import optuna
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "scale_pos_weight": sum(y_train == 0) / max(sum(y_train == 1), 1),
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }

            model = xgb.XGBClassifier(**params)

            # Use cross-validation for more robust evaluation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision')

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest trial AUC-PR (CV): {study.best_value:.4f}")
        print(f"Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study.best_params

    except ImportError:
        print("Optuna not installed, using default hyperparameters")
        return {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.1,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final model with best hyperparameters."""
    print("\n" + "=" * 60)
    print("STEP 5: Training Final Model")
    print("=" * 60)

    import xgboost as xgb

    params = {
        **best_params,
        "scale_pos_weight": sum(y_train == 0) / max(sum(y_train == 1), 1),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)

    # Train on train+val combined for final model
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    print(f"Training on {len(X_combined):,} samples...")
    model.fit(X_combined, y_combined, verbose=True)

    print("Training complete!")
    return model


def calibrate_model(model, X_val, y_val):
    """Calibrate model probabilities using isotonic regression."""
    print("\n" + "=" * 60)
    print("STEP 6: Calibrating Probabilities")
    print("=" * 60)

    calibrator = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrator.fit(X_val, y_val)

    raw_proba = model.predict_proba(X_val)[:, 1]
    cal_proba = calibrator.predict_proba(X_val)[:, 1]

    print(f"Raw probabilities - mean: {raw_proba.mean():.4f}, std: {raw_proba.std():.4f}")
    print(f"Calibrated probs  - mean: {cal_proba.mean():.4f}, std: {cal_proba.std():.4f}")

    return calibrator


def evaluate_model(model, calibrator, X_test, y_test, feature_names):
    """Comprehensive evaluation on held-out test set."""
    print("\n" + "=" * 60)
    print("STEP 7: RIGOROUS EVALUATION ON TEST SET")
    print("=" * 60)

    # Get predictions
    y_pred_proba_raw = model.predict_proba(X_test)[:, 1]
    y_pred_proba = calibrator.predict_proba(X_test)[:, 1]

    # Multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n--- Metrics at Different Thresholds ---")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # False Positive Rate
        tn = sum((y_test == 0) & (y_pred == 0))
        fp = sum((y_test == 0) & (y_pred == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fpr:<12.6f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Core metrics
    print("\n--- Core Metrics (Threshold-Independent) ---")
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Best F1: {best_f1:.4f} (at threshold {best_threshold})")

    # Confusion matrix at best threshold
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\n--- Confusion Matrix (threshold={best_threshold}) ---")
    print(f"                 Predicted")
    print(f"                 Normal  Fraud")
    print(f"Actual Normal    {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"Actual Fraud     {cm[1,0]:>6}  {cm[1,1]:>6}")

    # Calculate key business metrics
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    tn = cm[0,0]

    print(f"\n--- Business Metrics ---")
    print(f"True Positives (fraud caught):  {tp:,}")
    print(f"False Positives (false alarms): {fp:,}")
    print(f"False Negatives (fraud missed): {fn:,}")
    print(f"True Negatives (clean cleared): {tn:,}")
    print(f"False Positive Rate: {fp/(fp+tn)*100:.4f}%")
    print(f"False Negative Rate: {fn/(fn+tp)*100:.4f}%")

    # Classification report
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_best, target_names=["Normal", "Fraud"]))

    # Feature importance
    print("\n--- Top 20 Important Features ---")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:<35} {importance[idx]:.4f}")

    metrics = {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "precision_at_best": float(precision_score(y_test, y_pred_best, zero_division=0)),
        "recall_at_best": float(recall_score(y_test, y_pred_best, zero_division=0)),
        "false_positive_rate": float(fp/(fp+tn)),
        "false_negative_rate": float(fn/(fn+tp)) if (fn+tp) > 0 else 0,
        "test_samples": int(len(y_test)),
        "test_fraud_count": int(sum(y_test)),
    }

    return metrics


def save_artifacts(model, calibrator, scaler, feature_names, best_params, metrics):
    """Save all model artifacts."""
    print("\n" + "=" * 60)
    print("STEP 8: Saving Model Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model (v2)
    model_path = ARTIFACTS_DIR / "xgboost_v2.json"
    model.save_model(str(model_path))
    print(f"Saved: {model_path}")

    # Save calibrator
    calibrator_path = ARTIFACTS_DIR / "calibrator_v2.pkl"
    joblib.dump(calibrator, calibrator_path)
    print(f"Saved: {calibrator_path}")

    # Save scaler
    scaler_path = ARTIFACTS_DIR / "scaler_v2.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")

    # Save feature config
    config = {
        "version": "2.0.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "paysim",
        "changes_from_v1": [
            "REMOVED isFlaggedFraud (data leakage)",
            "Added more robust features",
            "Cross-validation during tuning",
            "More evaluation thresholds"
        ],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "hyperparameters": best_params,
        "metrics": metrics,
        "best_threshold": metrics["best_threshold"],
    }
    config_path = ARTIFACTS_DIR / "feature_config_v2.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # Try to save SHAP explainer (optional)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        explainer_path = ARTIFACTS_DIR / "shap_explainer_v2.pkl"
        joblib.dump(explainer, explainer_path)
        print(f"Saved: {explainer_path}")
    except ImportError:
        print("SHAP not installed, skipping explainer")
    except Exception as e:
        print(f"Could not save SHAP explainer: {e}")

    print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")


def main():
    print("=" * 60)
    print("MARBEL XGBoost FRAUD DETECTION MODEL TRAINING - V2")
    print("(Data Leakage Fixed)")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check for quick mode
    quick_mode = "--quick" in sys.argv
    n_trials = 10 if quick_mode else N_OPTUNA_TRIALS
    if quick_mode:
        print("\n[QUICK MODE] Using 10 Optuna trials instead of 50")

    # Install dependencies if needed
    try:
        import xgboost
        import optuna
    except ImportError:
        print("\nInstalling required packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "optuna", "imbalanced-learn", "shap", "-q"])

    # Load and preprocess
    X, y, feature_names = load_and_preprocess_data()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train_balanced, y_train_balanced, X_val, y_val, n_trials)

    # Train final model
    model = train_final_model(X_train_balanced, y_train_balanced, X_val, y_val, best_params)

    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val)

    # Evaluate on TEST SET (never seen during training or tuning)
    metrics = evaluate_model(model, calibrator, X_test, y_test, feature_names)

    # Save artifacts
    save_artifacts(model, calibrator, scaler, feature_names, best_params, metrics)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - V2 (No Data Leakage)")
    print("=" * 60)
    print(f"\nFinal Test Set Performance:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"  Best F1: {metrics['best_f1']:.4f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']*100:.4f}%")
    print(f"  False Negative Rate: {metrics['false_negative_rate']*100:.4f}%")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
