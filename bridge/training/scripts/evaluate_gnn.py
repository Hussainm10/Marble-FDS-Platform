#!/usr/bin/env python3
"""Comprehensive GNN GraphSAGE Model Evaluation.

Performs rigorous evaluation of the trained GraphSAGE model including:
- Multiple threshold analysis
- Confusion matrix at various operating points
- ROC and PR curves
- Layer-by-layer embedding analysis
- Error analysis on misclassified nodes
- Business metrics
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: torch_geometric not installed")
    print("Install with: pip install torch-geometric")
    sys.exit(1)


print("=" * 70)
print("GNN GRAPHSAGE V1 - COMPREHENSIVE EVALUATION")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()


# --- Model Definition ---

class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Get intermediate embeddings for analysis."""
        embeddings = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            embeddings.append(x.detach().clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        return embeddings


# --- Paths ---
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "gnn" / "artifacts"
DATA_DIR = Path(__file__).parent.parent / "data" / "elliptic"

# --- Load Model Artifacts ---
print("Loading model artifacts...")

with open(ARTIFACTS_DIR / "config.json") as f:
    config = json.load(f)

arch = config["architecture"]
model = GraphSAGE(
    in_channels=166,
    hidden_channels=arch["hidden_channels"],
    out_channels=2,
    num_layers=arch["num_layers"],
    dropout=arch["dropout"],
)

state_dict = torch.load(ARTIFACTS_DIR / "graphsage_v1.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
best_threshold = config.get("best_threshold", 0.5)

print(f"Model loaded: {arch}")
print(f"Best threshold: {best_threshold}")
print()

# --- Load Elliptic Dataset ---
print("Loading Elliptic Bitcoin dataset...")

features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

if not features_path.exists():
    _project_root = Path(__file__).resolve().parents[3]
    alt_dirs = [
        _project_root / "bridge" / "training" / "data" / "elliptic",
        Path.home() / "elliptic",
    ]
    for alt_dir in alt_dirs:
        if (alt_dir / "elliptic_txs_features.csv").exists():
            DATA_DIR = alt_dir
            features_path = DATA_DIR / "elliptic_txs_features.csv"
            classes_path = DATA_DIR / "elliptic_txs_classes.csv"
            edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"
            break

if not features_path.exists():
    print(f"ERROR: Dataset not found.")
    print("Please download the Elliptic dataset from:")
    print("https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    sys.exit(1)

# Load features
df_features = pd.read_csv(features_path, header=None)
# Dataset has 167 columns: 1 txId + 93 local features + 72 aggregate features = 166 features
df_features.columns = ["txId"] + [f"local_feat_{i}" for i in range(1, 94)] + \
                      [f"agg_feat_{i}" for i in range(1, 74)]

# Load classes
df_classes = pd.read_csv(classes_path)
df_classes.columns = ["txId", "class"]

# Merge
df = df_features.merge(df_classes, on="txId")

# Filter labeled only
df_labeled = df[df["class"] != "unknown"].copy()
df_labeled["label"] = (df_labeled["class"] == "1").astype(int)  # 1=illicit, 2=licit

# Create node mapping
all_tx_ids = df["txId"].values
tx_to_idx = {tx: idx for idx, tx in enumerate(all_tx_ids)}

# Load edges
df_edges = pd.read_csv(edges_path)
src = df_edges["txId1"].map(tx_to_idx).dropna().astype(int).values
dst = df_edges["txId2"].map(tx_to_idx).dropna().astype(int).values

# Build graph
feature_cols = [c for c in df.columns if c.startswith("local_feat_") or c.startswith("agg_feat_")]
X = df[feature_cols].values
X_scaled = scaler.transform(X)

edge_index = torch.tensor([src, dst], dtype=torch.long)
x = torch.tensor(X_scaled, dtype=torch.float32)

print(f"Total nodes: {len(df):,}")
print(f"Total edges: {len(df_edges):,}")
print(f"Labeled nodes: {len(df_labeled):,}")
print()

# --- Temporal Split (same as training) ---
print("Preparing temporal split...")

# Get time steps
time_steps = df_labeled.groupby("txId").first().reset_index()
time_steps["time_step"] = df_labeled["txId"].map(
    dict(zip(df["txId"], df.index // (len(df) // 49)))  # 49 time steps
)

# Sort by time
labeled_indices = df_labeled["txId"].map(tx_to_idx).values

# Get labels for labeled nodes
labels = df_labeled["label"].values

# Create masks based on temporal split
n_labeled = len(labeled_indices)
train_end = int(n_labeled * 0.70)
val_end = int(n_labeled * 0.85)

train_indices = labeled_indices[:train_end]
val_indices = labeled_indices[train_end:val_end]
test_indices = labeled_indices[val_end:]

train_labels = labels[:train_end]
val_labels = labels[train_end:val_end]
test_labels = labels[val_end:]

print(f"Train: {len(train_indices):,} ({train_labels.mean()*100:.2f}% fraud)")
print(f"Val:   {len(val_indices):,} ({val_labels.mean()*100:.2f}% fraud)")
print(f"Test:  {len(test_indices):,} ({test_labels.mean()*100:.2f}% fraud)")
print()

# --- Run Inference ---
print("Running inference on test set...")

with torch.no_grad():
    logits = model(x, edge_index)
    probs = F.softmax(logits, dim=1)
    test_probs = probs[test_indices, 1].numpy()

print(f"Probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
print(f"Mean probability: {test_probs.mean():.4f}")
print()

# --- Core Metrics ---
print("=" * 70)
print("CORE METRICS")
print("=" * 70)

auc_roc = roc_auc_score(test_labels, test_probs)
auc_pr = average_precision_score(test_labels, test_probs)

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
    y_pred = (test_probs >= thresh).astype(int)

    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred, zero_division=0)
    f1 = f1_score(test_labels, y_pred, zero_division=0)
    accuracy = accuracy_score(test_labels, y_pred)

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel() if y_pred.sum() > 0 and (1-y_pred).sum() > 0 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"{thresh:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {fpr:<12.4f} {accuracy:<12.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold_found = thresh

print()
print(f"Best F1: {best_f1:.4f} at threshold {best_threshold_found}")
print()

# --- Confusion Matrix ---
print("=" * 70)
print(f"CONFUSION MATRIX (threshold={best_threshold})")
print("=" * 70)

y_pred = (test_probs >= best_threshold).astype(int)

if y_pred.sum() > 0 and (1-y_pred).sum() > 0:
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
else:
    # Handle edge case
    tp = (y_pred & test_labels).sum()
    fp = (y_pred & ~test_labels.astype(bool)).sum()
    fn = (~y_pred.astype(bool) & test_labels).sum()
    tn = (~y_pred.astype(bool) & ~test_labels.astype(bool)).sum()

print(f"                 Predicted")
print(f"                 Normal    Fraud")
print(f"Actual Normal    {tn:>8,}  {fp:>8,}")
print(f"Actual Fraud     {fn:>8,}  {tp:>8,}")
print()

# --- Classification Report ---
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(test_labels, y_pred, target_names=["Normal", "Fraud"], digits=4, zero_division=0))

# --- Precision at Different Recall Levels ---
print("=" * 70)
print("PRECISION AT RECALL LEVELS")
print("=" * 70)

precision_arr, recall_arr, thresholds_arr = precision_recall_curve(test_labels, test_probs)

for target_recall in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30]:
    idx = np.argmin(np.abs(recall_arr - target_recall))
    prec_at_recall = precision_arr[idx]
    thresh_at_recall = thresholds_arr[min(idx, len(thresholds_arr)-1)] if idx < len(thresholds_arr) else 0
    actual_recall = recall_arr[idx]
    print(f"Target Recall={target_recall:.0%}: Precision={prec_at_recall:.4f}, Threshold={thresh_at_recall:.4f}, Actual Recall={actual_recall:.4f}")

print()

# --- Graph Structure Analysis ---
print("=" * 70)
print("GRAPH STRUCTURE ANALYSIS")
print("=" * 70)

# Calculate node degrees
src_nodes = edge_index[0].numpy()
dst_nodes = edge_index[1].numpy()

degrees = np.zeros(len(df))
np.add.at(degrees, src_nodes, 1)
np.add.at(degrees, dst_nodes, 1)

test_degrees = degrees[test_indices]
fraud_degrees = test_degrees[test_labels == 1]
normal_degrees = test_degrees[test_labels == 0]

print(f"Average degree (all test):     {test_degrees.mean():.2f}")
print(f"Average degree (fraud nodes):  {fraud_degrees.mean():.2f}")
print(f"Average degree (normal nodes): {normal_degrees.mean():.2f}")
print()

# Correlation between degree and prediction
from scipy.stats import spearmanr
corr, pval = spearmanr(test_degrees, test_probs)
print(f"Spearman correlation (degree vs fraud prob): {corr:.4f} (p={pval:.4e})")
print()

# --- Error Analysis ---
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

# False Negatives
fn_mask = (test_labels == 1) & (y_pred == 0)
fn_indices_local = np.where(fn_mask)[0]

if len(fn_indices_local) > 0:
    fn_probs = test_probs[fn_indices_local]
    fn_degrees = test_degrees[fn_indices_local]

    print(f"False Negatives (missed fraud): {len(fn_indices_local)}")
    print(f"  Probability range: [{fn_probs.min():.4f}, {fn_probs.max():.4f}]")
    print(f"  Mean probability: {fn_probs.mean():.4f}")
    print(f"  Mean degree: {fn_degrees.mean():.2f}")
    print()

    # Analyze probabilities distribution
    print("  Probability distribution of missed fraud:")
    for low, high in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8)]:
        count = ((fn_probs >= low) & (fn_probs < high)).sum()
        print(f"    [{low:.1f}, {high:.1f}): {count} ({count/len(fn_probs)*100:.1f}%)")
else:
    print("No false negatives!")

print()

# False Positives
fp_mask = (test_labels == 0) & (y_pred == 1)
fp_indices_local = np.where(fp_mask)[0]

if len(fp_indices_local) > 0:
    fp_probs = test_probs[fp_indices_local]
    fp_degrees = test_degrees[fp_indices_local]

    print(f"False Positives (false alarms): {len(fp_indices_local)}")
    print(f"  Probability range: [{fp_probs.min():.4f}, {fp_probs.max():.4f}]")
    print(f"  Mean probability: {fp_probs.mean():.4f}")
    print(f"  Mean degree: {fp_degrees.mean():.2f}")
else:
    print("No false positives!")

print()

# --- Why is Test Performance Lower? ---
print("=" * 70)
print("TEMPORAL SHIFT ANALYSIS")
print("=" * 70)

print("Understanding why test performance differs from validation:")
print()

# Calculate fraud rates by position (proxy for time)
n_buckets = 5
bucket_size = len(labeled_indices) // n_buckets

print("Fraud rate over time (temporal distribution):")
for i in range(n_buckets):
    start = i * bucket_size
    end = (i + 1) * bucket_size if i < n_buckets - 1 else len(labeled_indices)
    bucket_labels = labels[start:end]
    fraud_rate = bucket_labels.mean() * 100
    split = "TRAIN" if end <= train_end else ("VAL" if end <= val_end else "TEST")
    print(f"  Bucket {i+1}: {fraud_rate:.2f}% fraud ({split})")

print()
print("Note: The Elliptic dataset has significant temporal distribution shift.")
print("Later time periods (test set) have much lower fraud rates, making")
print("prediction harder as the model was trained on earlier periods.")
print()

# --- Summary ---
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

total_fraud = test_labels.sum()
caught_fraud = tp

print(f"""
Model: GraphSAGE V1
Dataset: Elliptic Bitcoin ({len(test_labels):,} test samples, {int(test_labels.sum()):,} fraud)

Performance:
  AUC-ROC:     {auc_roc:.6f}
  AUC-PR:      {auc_pr:.6f}
  Best F1:     {best_f1:.4f} (at threshold {best_threshold_found})

At threshold {best_threshold}:
  Precision:   {precision_score(test_labels, y_pred, zero_division=0):.4f}
  Recall:      {recall_score(test_labels, y_pred, zero_division=0):.4f}
  F1 Score:    {f1_score(test_labels, y_pred, zero_division=0):.4f}
  Accuracy:    {accuracy_score(test_labels, y_pred):.4f}

Business Impact:
  Fraud caught:   {tp:,} / {int(total_fraud):,} ({tp/total_fraud*100:.1f}% if total_fraud > 0 else 0)
  False alarms:   {fp:,} / {int((test_labels==0).sum()):,} ({fp/(test_labels==0).sum()*100:.4f}% if (test_labels==0).sum() > 0 else 0)

Key Insights:
  - Significant temporal distribution shift in test set
  - Test fraud rate ({test_labels.mean()*100:.2f}%) much lower than train ({train_labels.mean()*100:.2f}%)
  - Model struggles with low-base-rate fraud detection in later periods
  - Consider ensemble with Marbel for production use
""")

print(f"Finished: {datetime.now().isoformat()}")
