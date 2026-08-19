#!/usr/bin/env python3
"""Comprehensive Marbel XGBoost V2 Model Evaluation.

Performs rigorous evaluation of the trained XGBoost model including:
- Multiple threshold analysis
- Confusion matrix at various operating points
- ROC and PR curves
- Feature importance analysis
- Error analysis on misclassified samples
- Business metrics (cost-benefit analysis)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

print("=" * 70)
print("MARBEL XGBOOST V2 - COMPREHENSIVE EVALUATION")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print()

# --- Paths ---
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "marbel" / "artifacts"
DATA_DIR = Path(__file__).parent.parent / "data" / "paysim"

# --- Load Model Artifacts ---
print("Loading model artifacts...")

model = xgb.XGBClassifier()
model.load_model(ARTIFACTS_DIR / "xgboost_v2.json")

scaler = joblib.load(ARTIFACTS_DIR / "scaler_v2.pkl")
calibrator = joblib.load(ARTIFACTS_DIR / "calibrator_v2.pkl")

with open(ARTIFACTS_DIR / "feature_config_v2.json") as f:
    config = json.load(f)

feature_names = config["feature_names"]
best_threshold = config["best_threshold"]
print(f"Model loaded: {config['feature_count']} features, threshold={best_threshold}")
print()

# --- Load Data ---
print("Loading PaySim dataset...")

csv_path = DATA_DIR / "paysim_transactions.csv"
if not csv_path.exists():
    # Try alternative paths
    _project_root = Path(__file__).resolve().parents[3]
    alt_paths = [
        _project_root / "bridge" / "training" / "data" / "paysim" / "paysim_transactions.csv",
        _project_root / "bridge" / "training" / "data" / "paysim" / "PS_20174392719_1491204439457_log.csv",
        Path.home() / "paysim_transactions.csv",
    ]
    for alt in alt_paths:
        if alt.exists():
            csv_path = alt
            break

if not csv_path.exists():
    print(f"ERROR: Dataset not found. Please place paysim_transactions.csv in {DATA_DIR}")
    print("You can download it from: https://www.kaggle.com/datasets/ealaxi/paysim1")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} transactions")

# --- Feature Engineering (must match training) ---
print("\nEngineering features...")

# Amount features
df["amount_log"] = np.log1p(df["amount"])
df["amount_round_100"] = (df["amount"] % 100 == 0).astype(int)
df["amount_round_1000"] = (df["amount"] % 1000 == 0).astype(int)
df["amount_round_10000"] = (df["amount"] % 10000 == 0).astype(int)

amount_median = df["amount"].median()
amount_p90 = df["amount"].quantile(0.90)
amount_p99 = df["amount"].quantile(0.99)

df["amount_above_median"] = (df["amount"] > amount_median).astype(int)
df["amount_above_p90"] = (df["amount"] > amount_p90).astype(int)
df["amount_above_p99"] = (df["amount"] > amount_p99).astype(int)

# Balance features
df["balance_orig_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["balance_dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["amount_to_dest_balance_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)
df["balance_orig_change_pct"] = df["balance_orig_delta"] / (df["oldbalanceOrg"] + 1)
df["balance_dest_change_pct"] = df["balance_dest_delta"] / (df["oldbalanceDest"] + 1)

# Suspicious patterns
df["orig_zeroed_out"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
df["exact_amount_transfer"] = (
    (abs(df["balance_orig_delta"]) == df["amount"]) |
    (abs(df["balance_dest_delta"]) == df["amount"])
).astype(int)
df["dest_new_account"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] > 0)).astype(int)
df["full_balance_withdrawal"] = ((df["amount"] >= df["oldbalanceOrg"] * 0.99) & (df["oldbalanceOrg"] > 0)).astype(int)
df["dest_receives_more"] = (df["balance_dest_delta"] > df["amount"]).astype(int)
df["orig_balance_mismatch"] = (abs(df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]) > 1).astype(int)

# Time features
df["hour_of_day"] = df["step"] % 24
df["day_of_week"] = (df["step"] // 24) % 7
df["day_of_month"] = ((df["step"] // 24) % 30) + 1
df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_early_morning"] = ((df["hour_of_day"] >= 2) & (df["hour_of_day"] <= 5)).astype(int)

# Transaction type encoding
df = pd.get_dummies(df, columns=["type"], prefix="type")

# Ensure all type columns exist
for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
    if col not in df.columns:
        df[col] = 0

# Select features
X = df[feature_names].values
y = df["isFraud"].values

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")
print(f"Fraud rate: {y.mean()*100:.4f}%")
print()

# --- Train/Test Split (temporal, same as training) ---
print("Splitting data (80/10/10 temporal split)...")

n = len(df)
train_idx = int(n * 0.8)
val_idx = int(n * 0.9)

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

print(f"Train: {len(y_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"Val:   {len(y_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test:  {len(y_test):,} ({y_test.mean()*100:.2f}% fraud)")
print()

# --- Scale Features ---
X_test_scaled = scaler.transform(X_test)

# --- Get Predictions ---
print("Running inference on test set...")

y_proba_raw = model.predict_proba(X_test_scaled)[:, 1]
# Calibrator wraps the model and expects features, not probabilities
y_proba_calibrated = calibrator.predict_proba(X_test_scaled)[:, 1]

print(f"Raw probability range: [{y_proba_raw.min():.4f}, {y_proba_raw.max():.4f}]")
print(f"Calibrated probability range: [{y_proba_calibrated.min():.4f}, {y_proba_calibrated.max():.4f}]")
print()

# --- Core Metrics ---
print("=" * 70)
print("CORE METRICS")
print("=" * 70)

auc_roc = roc_auc_score(y_test, y_proba_calibrated)
auc_pr = average_precision_score(y_test, y_proba_calibrated)

print(f"AUC-ROC:  {auc_roc:.6f}")
print(f"AUC-PR:   {auc_pr:.6f}")
print()

# --- Threshold Analysis ---
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12} {'Accuracy':<12}")
print("-" * 72)

best_f1 = 0
best_threshold_found = 0.5

for thresh in thresholds:
    y_pred = (y_proba_calibrated >= thresh).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"{thresh:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {fpr:<12.4f} {accuracy:<12.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold_found = thresh

print()
print(f"Best F1: {best_f1:.4f} at threshold {best_threshold_found}")
print()

# --- Confusion Matrix at Best Threshold ---
print("=" * 70)
print(f"CONFUSION MATRIX (threshold={best_threshold})")
print("=" * 70)

y_pred = (y_proba_calibrated >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"                 Predicted")
print(f"                 Normal    Fraud")
print(f"Actual Normal    {tn:>8,}  {fp:>8,}")
print(f"Actual Fraud     {fn:>8,}  {tp:>8,}")
print()

# --- Classification Report ---
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], digits=4))

# --- Business Metrics ---
print("=" * 70)
print("BUSINESS METRICS")
print("=" * 70)

# Assume: Cost of missing fraud = $1000, Cost of false alarm = $10
cost_fn = 1000  # Cost of false negative (missed fraud)
cost_fp = 10    # Cost of false positive (investigation cost)

total_cost_no_model = y_test.sum() * cost_fn  # All fraud missed
total_cost_with_model = fn * cost_fn + fp * cost_fp

savings = total_cost_no_model - total_cost_with_model
savings_pct = savings / total_cost_no_model * 100 if total_cost_no_model > 0 else 0

print(f"True Positives (fraud caught):    {tp:>8,}")
print(f"False Positives (false alarms):   {fp:>8,}")
print(f"False Negatives (fraud missed):   {fn:>8,}")
print(f"True Negatives (correct normal):  {tn:>8,}")
print()
print(f"Cost assumptions: FN=${cost_fn}, FP=${cost_fp}")
print(f"Cost without model: ${total_cost_no_model:,.0f}")
print(f"Cost with model:    ${total_cost_with_model:,.0f}")
print(f"Savings:            ${savings:,.0f} ({savings_pct:.1f}%)")
print()

# --- Precision at Different Recall Levels ---
print("=" * 70)
print("PRECISION AT RECALL LEVELS")
print("=" * 70)

precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test, y_proba_calibrated)

for target_recall in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]:
    idx = np.argmin(np.abs(recall_arr - target_recall))
    prec_at_recall = precision_arr[idx]
    thresh_at_recall = thresholds_arr[min(idx, len(thresholds_arr)-1)]
    print(f"Recall={target_recall:.0%}: Precision={prec_at_recall:.4f}, Threshold={thresh_at_recall:.4f}")

print()

# --- Feature Importance ---
print("=" * 70)
print("TOP 15 FEATURE IMPORTANCE (Gain)")
print("=" * 70)

importance = model.get_booster().get_score(importance_type="gain")
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for i, (feat, gain) in enumerate(importance_sorted[:15], 1):
    # Map feature index to name
    feat_idx = int(feat.replace("f", ""))
    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else feat
    print(f"{i:>2}. {feat_name:<35} {gain:>12.2f}")

print()

# --- Error Analysis ---
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

# False Negatives (missed fraud)
fn_mask = (y_test == 1) & (y_pred == 0)
fn_indices = np.where(fn_mask)[0]

if len(fn_indices) > 0:
    fn_probas = y_proba_calibrated[fn_indices]
    print(f"False Negatives (missed fraud): {len(fn_indices)}")
    print(f"  Probability range: [{fn_probas.min():.4f}, {fn_probas.max():.4f}]")
    print(f"  Mean probability:  {fn_probas.mean():.4f}")
    print(f"  Median probability: {np.median(fn_probas):.4f}")

    # Analyze characteristics of missed fraud
    fn_data = X_test[fn_indices]
    print(f"\n  Characteristics of missed fraud:")
    for i, feat_name in enumerate(feature_names[:10]):
        mean_val = fn_data[:, i].mean()
        overall_mean = X_test[:, i].mean()
        if overall_mean != 0:
            ratio = mean_val / overall_mean
            print(f"    {feat_name:<30}: {mean_val:.2f} (vs overall {overall_mean:.2f}, ratio={ratio:.2f})")
else:
    print("No false negatives!")

print()

# False Positives (false alarms)
fp_mask = (y_test == 0) & (y_pred == 1)
fp_indices = np.where(fp_mask)[0]

if len(fp_indices) > 0:
    fp_probas = y_proba_calibrated[fp_indices]
    print(f"False Positives (false alarms): {len(fp_indices)}")
    print(f"  Probability range: [{fp_probas.min():.4f}, {fp_probas.max():.4f}]")
    print(f"  Mean probability:  {fp_probas.mean():.4f}")
else:
    print("No false positives!")

print()

# --- Summary ---
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(f"""
Model: XGBoost V2 (No data leakage)
Dataset: PaySim ({len(y_test):,} test samples, {y_test.sum():,} fraud)

Performance:
  AUC-ROC:     {auc_roc:.6f}
  AUC-PR:      {auc_pr:.6f}
  Best F1:     {best_f1:.4f} (at threshold {best_threshold_found})

At threshold {best_threshold}:
  Precision:   {precision_score(y_test, y_pred):.4f}
  Recall:      {recall_score(y_test, y_pred):.4f}
  F1 Score:    {f1_score(y_test, y_pred):.4f}
  Accuracy:    {accuracy_score(y_test, y_pred):.4f}

Business Impact:
  Fraud caught:   {tp:,} / {y_test.sum():,} ({tp/y_test.sum()*100:.1f}%)
  False alarms:   {fp:,} / {(y_test==0).sum():,} ({fp/(y_test==0).sum()*100:.4f}%)
  Cost savings:   ${savings:,.0f} ({savings_pct:.1f}%)
""")

print(f"Finished: {datetime.now().isoformat()}")
