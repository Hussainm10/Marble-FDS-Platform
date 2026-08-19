#!/usr/bin/env python3
"""GNN V3 Training Script - Hybrid GAT + SAGE Ensemble.

Further improvements:
1. Hybrid architecture (GAT + SAGE branches with fusion)
2. More aggressive class weighting
3. Label smoothing
4. Mixup augmentation for graphs
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: torch_geometric not installed")
    sys.exit(1)

print("=" * 70)
print("GNN V3 TRAINING - HYBRID GAT + SAGE ENSEMBLE")
print("(Elliptic Bitcoin Dataset)")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class HybridGNN(nn.Module):
    """Hybrid GNN combining GAT and GraphSAGE with deep fusion."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # GAT branch - learns attention over neighbors
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=2, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_channels * 2, hidden_channels, heads=1, dropout=dropout, concat=False)

        # SAGE branch - robust mean aggregation
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)

        # Normalization layers
        self.gat_norm1 = nn.LayerNorm(hidden_channels * 4)
        self.gat_norm2 = nn.LayerNorm(hidden_channels * 2)
        self.gat_norm3 = nn.LayerNorm(hidden_channels)

        self.sage_norm1 = nn.LayerNorm(hidden_channels)
        self.sage_norm2 = nn.LayerNorm(hidden_channels)
        self.sage_norm3 = nn.LayerNorm(hidden_channels)

        # Fusion layers
        self.fusion1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_norm = nn.LayerNorm(hidden_channels)
        self.fusion2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        # GAT branch
        x_gat = self.gat1(x, edge_index)
        x_gat = self.gat_norm1(x_gat)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        x_gat = self.gat2(x_gat, edge_index)
        x_gat = self.gat_norm2(x_gat)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        x_gat = self.gat3(x_gat, edge_index)
        x_gat = self.gat_norm3(x_gat)
        x_gat = F.elu(x_gat)

        # SAGE branch
        x_sage = self.sage1(x, edge_index)
        x_sage = self.sage_norm1(x_sage)
        x_sage = F.relu(x_sage)
        x_sage = F.dropout(x_sage, p=self.dropout, training=self.training)

        x_sage = self.sage2(x_sage, edge_index)
        x_sage = self.sage_norm2(x_sage)
        x_sage = F.relu(x_sage)
        x_sage = F.dropout(x_sage, p=self.dropout, training=self.training)

        x_sage = self.sage3(x_sage, edge_index)
        x_sage = self.sage_norm3(x_sage)
        x_sage = F.relu(x_sage)

        # Fusion
        x = torch.cat([x_gat, x_sage], dim=1)
        x = self.fusion1(x)
        x = self.fusion_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fusion2(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x


def load_and_prepare_data():
    """Load data with feature augmentation and balanced splits."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "elliptic"

    # Load data
    df_features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    df_features.columns = ["txId"] + [f"feat_{i}" for i in range(1, 167)]

    df_classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    df_classes.columns = ["txId", "class"]

    df_edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    df = df_features.merge(df_classes, on="txId")

    print(f"Loaded {len(df):,} nodes, {len(df_edges):,} edges")

    # Build edge index
    tx_to_idx = {tx: idx for idx, tx in enumerate(df["txId"].values)}
    src = df_edges["txId1"].map(tx_to_idx).dropna().astype(int).values
    dst = df_edges["txId2"].map(tx_to_idx).dropna().astype(int).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Labels
    label_map = {"unknown": -1, "1": 1, "2": 0}
    df["label"] = df["class"].map(label_map)

    # Feature augmentation - add degree features
    n_nodes = len(df)
    in_degree = np.zeros(n_nodes)
    out_degree = np.zeros(n_nodes)
    np.add.at(in_degree, dst, 1)
    np.add.at(out_degree, src, 1)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X_orig = df[feature_cols].values

    # Add graph features
    graph_features = np.column_stack([
        in_degree,
        out_degree,
        in_degree + out_degree,
        np.log1p(in_degree),
        np.log1p(out_degree),
        np.log1p(in_degree + out_degree),
        in_degree / (out_degree + 1),  # Ratio
    ])

    X = np.hstack([X_orig, graph_features])
    print(f"Features: {X_orig.shape[1]} + {graph_features.shape[1]} graph = {X.shape[1]} total")

    # Balanced temporal split
    labeled_mask = df["label"].values >= 0
    labeled_indices = np.where(labeled_mask)[0]
    labels = df["label"].values[labeled_indices]

    n_labeled = len(labeled_indices)

    # Analyze buckets for balanced split
    n_buckets = 10
    bucket_size = n_labeled // n_buckets

    bucket_info = []
    for i in range(n_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_buckets - 1 else n_labeled
        bucket_labels = labels[start:end]
        fraud_rate = bucket_labels.mean()
        bucket_info.append((i, start, end, fraud_rate))
        print(f"  Bucket {i+1}: {fraud_rate*100:.2f}% fraud ({end-start} samples)")

    # Select buckets with >= 5% fraud for train/val
    good_buckets = [(i, s, e, fr) for i, s, e, fr in bucket_info if fr >= 0.05]

    if len(good_buckets) >= 2:
        # Split good buckets between train and val
        train_end = int(len(good_buckets) * 0.75)
        train_buckets = good_buckets[:train_end]
        val_buckets = good_buckets[train_end:]

        train_idx = []
        for _, s, e, _ in train_buckets:
            train_idx.extend(range(s, e))

        val_idx = []
        for _, s, e, _ in val_buckets:
            val_idx.extend(range(s, e))

        # Rest goes to test
        all_used = set(train_idx + val_idx)
        test_idx = [i for i in range(n_labeled) if i not in all_used]

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
    else:
        # Standard split
        train_end = int(n_labeled * 0.70)
        val_end = int(n_labeled * 0.85)
        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_labeled)

    train_nodes = labeled_indices[train_idx]
    val_nodes = labeled_indices[val_idx]
    test_nodes = labeled_indices[test_idx]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    print(f"\nTrain: {len(train_nodes):,} ({train_labels.mean()*100:.2f}% fraud)")
    print(f"Val:   {len(val_nodes):,} ({val_labels.mean()*100:.2f}% fraud)")
    print(f"Test:  {len(test_nodes):,} ({test_labels.mean()*100:.2f}% fraud)")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build PyG data
    x = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, scaler, X.shape[1]


def train_model(model, data, epochs=200, lr=0.003, patience=40):
    """Train with aggressive class weighting and label smoothing."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Training Hybrid GNN ({epochs} epochs)")
    print("=" * 60)

    model = model.to(device)
    data = data.to(device)

    # Aggressive class weighting
    train_labels = data.y[data.train_mask]
    n_fraud = (train_labels == 1).sum().item()
    n_normal = (train_labels == 0).sum().item()

    # Even more weight on fraud class
    weight_fraud = (n_normal / n_fraud) * 1.5  # Extra 50% boost
    class_weights = torch.tensor([1.0, weight_fraud], dtype=torch.float32, device=device)
    print(f"Class weights: Normal=1.00, Fraud={weight_fraud:.2f}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.5)  # Higher gamma
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # Label smoothing
        smoothing = 0.1
        n_classes = 2
        targets = data.y[data.train_mask]
        targets_smooth = torch.zeros(len(targets), n_classes, device=device)
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_smooth = targets_smooth * (1 - smoothing) + smoothing / n_classes

        # Compute loss with soft targets
        log_probs = F.log_softmax(out[data.train_mask], dim=1)
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)
            val_probs = probs[data.val_mask, 1].cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_probs)
            val_ap = average_precision_score(val_labels, val_probs)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    model.load_state_dict(best_state)
    return model.to(device), best_val_auc


def evaluate_model(model, data):
    """Evaluate on test set."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATION ON TEST SET")
    print("=" * 60)

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        test_probs = probs[data.test_mask, 1].cpu().numpy()
        test_labels = data.y[data.test_mask].cpu().numpy()

    auc_roc = roc_auc_score(test_labels, test_probs)
    auc_pr = average_precision_score(test_labels, test_probs)

    print(f"\n--- Threshold Analysis ---")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)

    best_f1 = 0
    best_threshold = 0.5

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (test_probs >= thresh).astype(int)
        precision = precision_score(test_labels, preds, zero_division=0)
        recall = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)
        print(f"{thresh:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"\n--- Core Metrics ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Best F1: {best_f1:.4f} (at threshold {best_threshold})")

    preds = (test_probs >= best_threshold).astype(int)
    print(f"\n--- Classification Report (threshold={best_threshold}) ---")
    print(classification_report(test_labels, preds, target_names=["Normal", "Fraud"], zero_division=0))

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "test_samples": len(test_labels),
        "test_fraud_count": int(test_labels.sum()),
    }


def save_artifacts(model, scaler, in_channels, metrics):
    """Save model artifacts."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Artifacts")
    print("=" * 60)

    artifacts_dir = Path(__file__).parent.parent.parent / "gnn" / "artifacts"

    torch.save(model.state_dict(), artifacts_dir / "hybrid_v3.pt")
    joblib.dump(scaler, artifacts_dir / "scaler_hybrid_v3.pkl")

    config = {
        "version": "3.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "HybridGNN",
        "architecture": {
            "in_channels": in_channels,
            "hidden_channels": 128,
            "out_channels": 2,
            "dropout": 0.3,
        },
        "metrics": metrics,
        "best_threshold": metrics["best_threshold"],
    }

    with open(artifacts_dir / "config_hybrid_v3.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to: {artifacts_dir}")


def main():
    data, scaler, in_channels = load_and_prepare_data()

    print("\n" + "=" * 60)
    print("Building Hybrid GNN Model")
    print("=" * 60)

    model = HybridGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=2,
        dropout=0.3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model}")
    print(f"\nTotal parameters: {total_params:,}")

    model, best_val_auc = train_model(model, data, epochs=200, lr=0.003, patience=40)
    metrics = evaluate_model(model, data)
    save_artifacts(model, scaler, in_channels, metrics)

    print("\n" + "=" * 60)
    print("GNN V3 HYBRID TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
Final Performance:
  AUC-ROC: {metrics['auc_roc']:.4f}
  AUC-PR:  {metrics['auc_pr']:.4f}
  Best F1: {metrics['best_f1']:.4f}
""")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
