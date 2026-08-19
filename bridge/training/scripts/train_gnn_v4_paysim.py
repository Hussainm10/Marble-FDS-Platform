#!/usr/bin/env python3
"""GNN V4 Training Script - HybridGNN on PaySim Transaction Graph.

Trains GNN on PaySim dataset by constructing a transaction graph where:
- Nodes: Customer accounts (nameOrig and nameDest)
- Edges: Transactions between accounts
- Node features: Aggregated transaction statistics
- Edge labels: Fraud indicator

This makes the GNN model directly applicable to payment fraud detection.
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
    print("Run: pip install torch-geometric")
    sys.exit(1)

print("=" * 70)
print("GNN V4 TRAINING - HYBRID GNN ON PAYSIM TRANSACTION GRAPH")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "paysim"
PAYSIM_FILE = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "gnn" / "artifacts"
# Live monitoring outputs (separate from final artifacts so they're easy to tail/plot during training)
RUN_DIR = Path(__file__).parent.parent / "runs" / "gnn_v4_paysim"
METRICS_FILE = RUN_DIR / "metrics_history.json"
CURVE_FILE = RUN_DIR / "learning_curve.png"


def _save_learning_curve(history, path):
    """Render loss + val_auc + val_ap learning curves to a PNG. Best-effort; never raises."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    val_auc = [h["val_auc"] for h in history]
    val_ap = [h["val_ap"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, loss, label="train loss", color="#d62728")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Training loss"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(epochs, val_auc, label="val AUC-ROC", color="#1f77b4")
    ax2.plot(epochs, val_ap, label="val AUC-PR", color="#2ca02c")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_title("Validation metrics"); ax2.set_ylim(0, 1); ax2.grid(alpha=0.3); ax2.legend()
    fig.suptitle(f"GNN v4 (HybridGNN on PaySim) — epoch {epochs[-1]} / loss {loss[-1]:.4f} / val AUC {val_auc[-1]:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

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

        # GAT branch - learns attention over neighbors.
        # Reduced heads (4→2 in gat1) to keep full-graph activations under 16GB VRAM at ~440k-node graphs.
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels, heads=2, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_channels * 2, hidden_channels, heads=1, dropout=dropout, concat=False)

        # SAGE branch - robust mean aggregation
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)

        # Normalization layers
        self.gat_norm1 = nn.LayerNorm(hidden_channels * 2)
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


def load_and_build_graph(sample_size=None):
    """Load PaySim data and build transaction graph.

    Creates a node-level fraud detection task where:
    - Nodes are accounts (customers/merchants)
    - Edges are transactions
    - Node features are aggregated transaction statistics
    - Node labels indicate if the account was involved in fraud
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading PaySim Data and Building Graph")
    print("=" * 60)

    # Load data
    print(f"Loading from: {PAYSIM_FILE}")
    df = pd.read_csv(PAYSIM_FILE)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.3f}%")

    # Sample for faster training if needed
    if sample_size and sample_size < len(df):
        # Stratified sample to maintain fraud ratio
        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0]

        fraud_sample = min(len(fraud_df), int(sample_size * 0.1))  # Keep ~10% fraud
        normal_sample = sample_size - fraud_sample

        df = pd.concat([
            fraud_df.sample(n=fraud_sample, random_state=42),
            normal_df.sample(n=normal_sample, random_state=42)
        ]).reset_index(drop=True)
        print(f"Sampled to {len(df):,} transactions (fraud rate: {df['isFraud'].mean()*100:.3f}%)")

    # Get all unique accounts (both senders and receivers)
    all_accounts = pd.concat([df['nameOrig'], df['nameDest']]).unique()
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
    n_nodes = len(all_accounts)
    print(f"Unique accounts (nodes): {n_nodes:,}")

    # Build edge list (transactions)
    src_indices = df['nameOrig'].map(account_to_idx).values
    dst_indices = df['nameDest'].map(account_to_idx).values
    edge_index = torch.tensor(np.array([src_indices, dst_indices]), dtype=torch.long)
    print(f"Edges (transactions): {edge_index.shape[1]:,}")

    # Compute node features by aggregating transaction statistics
    print("\nComputing node features from transaction aggregations...")

    # Features for accounts as senders.
    # NOTE: isFraud aggregations were removed — they directly leak the node label
    # (which is built from isFraud below), making the task trivially solvable.
    sender_stats = df.groupby('nameOrig').agg({
        'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'oldbalanceOrg': ['mean', 'min'],
        'newbalanceOrig': ['mean', 'min'],
        'step': ['min', 'max', 'nunique'],
    }).fillna(0)
    sender_stats.columns = ['send_' + '_'.join(col) for col in sender_stats.columns]

    # Features for accounts as receivers (no isFraud — same leakage reason)
    receiver_stats = df.groupby('nameDest').agg({
        'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'oldbalanceDest': ['mean', 'min'],
        'newbalanceDest': ['mean', 'min'],
        'step': ['min', 'max', 'nunique'],
    }).fillna(0)
    receiver_stats.columns = ['recv_' + '_'.join(col) for col in receiver_stats.columns]

    # Transaction type counts as sender
    type_counts_send = pd.crosstab(df['nameOrig'], df['type'])
    type_counts_send.columns = ['send_type_' + col for col in type_counts_send.columns]

    # Transaction type counts as receiver
    type_counts_recv = pd.crosstab(df['nameDest'], df['type'])
    type_counts_recv.columns = ['recv_type_' + col for col in type_counts_recv.columns]

    # Create node feature matrix
    node_features = pd.DataFrame(index=all_accounts)
    node_features = node_features.join(sender_stats, how='left')
    node_features = node_features.join(receiver_stats, how='left')
    node_features = node_features.join(type_counts_send, how='left')
    node_features = node_features.join(type_counts_recv, how='left')
    node_features = node_features.fillna(0)

    # Add graph-based features
    in_degree = np.zeros(n_nodes)
    out_degree = np.zeros(n_nodes)
    np.add.at(in_degree, dst_indices, 1)
    np.add.at(out_degree, src_indices, 1)

    node_features['in_degree'] = in_degree
    node_features['out_degree'] = out_degree
    node_features['total_degree'] = in_degree + out_degree
    node_features['in_out_ratio'] = in_degree / (out_degree + 1)
    node_features['log_in_degree'] = np.log1p(in_degree)
    node_features['log_out_degree'] = np.log1p(out_degree)

    X = node_features.values.astype(np.float32)
    feature_names = list(node_features.columns)
    print(f"Node features: {X.shape[1]}")

    # Node labels: account is fraudulent if involved in ANY fraud transaction
    fraud_senders = set(df[df['isFraud'] == 1]['nameOrig'].unique())
    fraud_receivers = set(df[df['isFraud'] == 1]['nameDest'].unique())
    fraud_accounts = fraud_senders | fraud_receivers

    y = np.array([1 if acc in fraud_accounts else 0 for acc in all_accounts])
    print(f"Fraudulent accounts: {y.sum():,} ({y.mean()*100:.2f}%)")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create PyG Data object
    x = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split into train/val/test (temporal-ish: random but stratified)
    np.random.seed(42)
    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    # Stratified split
    fraud_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]
    np.random.shuffle(fraud_indices)
    np.random.shuffle(normal_indices)

    # 70% train, 15% val, 15% test
    train_fraud = fraud_indices[:int(0.7 * len(fraud_indices))]
    val_fraud = fraud_indices[int(0.7 * len(fraud_indices)):int(0.85 * len(fraud_indices))]
    test_fraud = fraud_indices[int(0.85 * len(fraud_indices)):]

    train_normal = normal_indices[:int(0.7 * len(normal_indices))]
    val_normal = normal_indices[int(0.7 * len(normal_indices)):int(0.85 * len(normal_indices))]
    test_normal = normal_indices[int(0.85 * len(normal_indices)):]

    train_idx = np.concatenate([train_fraud, train_normal])
    val_idx = np.concatenate([val_fraud, val_normal])
    test_idx = np.concatenate([test_fraud, test_normal])

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nTrain: {train_mask.sum().item():,} ({y[train_idx].mean()*100:.2f}% fraud)")
    print(f"Val:   {val_mask.sum().item():,} ({y[val_idx].mean()*100:.2f}% fraud)")
    print(f"Test:  {test_mask.sum().item():,} ({y[test_idx].mean()*100:.2f}% fraud)")

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data, scaler, feature_names


def train_model(model, data, epochs=150, lr=0.003, patience=30):
    """Train with class weighting and early stopping."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Training HybridGNN ({epochs} epochs)")
    print("=" * 60)

    model = model.to(device)
    data = data.to(device)

    # Aggressive class weighting for imbalanced data
    train_labels = data.y[data.train_mask]
    n_fraud = (train_labels == 1).sum().item()
    n_normal = (train_labels == 0).sum().item()

    weight_fraud = (n_normal / n_fraud) * 2.0  # Extra boost for fraud class
    class_weights = torch.tensor([1.0, weight_fraud], dtype=torch.float32, device=device)
    print(f"Class weights: Normal=1.00, Fraud={weight_fraud:.2f}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    best_val_auc = 0
    best_state = None
    patience_counter = 0
    metrics_history = []
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # Label smoothing
        smoothing = 0.1
        targets = data.y[data.train_mask]
        targets_smooth = torch.zeros(len(targets), 2, device=device)
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_smooth = targets_smooth * (1 - smoothing) + smoothing / 2

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

            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_probs)
                val_ap = average_precision_score(val_labels, val_probs)
            else:
                val_auc = 0.5
                val_ap = 0.0

        # Per-epoch live monitoring: print one line, dump metrics history JSON, refresh PNG every 5 epochs.
        is_best = val_auc > best_val_auc
        marker = " *" if is_best else ""
        print(
            f"[{epoch:>3}/{epochs}] loss={loss.item():.4f} val_auc={val_auc:.4f} "
            f"val_ap={val_ap:.4f} best_auc={max(best_val_auc, val_auc):.4f}"
            f" lr={scheduler.get_last_lr()[0]:.2e}{marker}",
            flush=True,
        )
        metrics_history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "lr": float(scheduler.get_last_lr()[0]),
            "best_auc": float(max(best_val_auc, val_auc)),
        })
        try:
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics_history, f, indent=2)
            if epoch % 5 == 0 or epoch == 1:
                _save_learning_curve(metrics_history, CURVE_FILE)
        except Exception as e:
            print(f"  [warn] could not write monitoring artifact: {e}", flush=True)

        if is_best:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final curve render after training completes
    _save_learning_curve(metrics_history, CURVE_FILE)

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

    if len(np.unique(test_labels)) < 2:
        print("WARNING: Test set has only one class, cannot compute AUC")
        return {"error": "single_class_test_set"}

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
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "test_samples": int(len(test_labels)),
        "test_fraud_count": int(test_labels.sum()),
    }


def save_artifacts(model, scaler, feature_names, metrics):
    """Save model artifacts."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), ARTIFACTS_DIR / "hybrid_v4_paysim.pt")
    print(f"Saved: hybrid_v4_paysim.pt")

    # Save scaler
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_hybrid_v4.pkl")
    print(f"Saved: scaler_hybrid_v4.pkl")

    # Save config
    config = {
        "version": "4.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "HybridGNN",
        "dataset": "PaySim",
        "architecture": {
            "in_channels": len(feature_names),
            "hidden_channels": 128,
            "out_channels": 2,
            "dropout": 0.3,
        },
        "feature_names": feature_names,
        "metrics": metrics,
        "best_threshold": metrics.get("best_threshold", 0.5),
    }

    with open(ARTIFACTS_DIR / "config_hybrid_v4.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: config_hybrid_v4.json")

    print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")


def main():
    # Check if data exists
    if not PAYSIM_FILE.exists():
        print(f"ERROR: PaySim data not found at {PAYSIM_FILE}")
        print("Please ensure the data is downloaded.")
        sys.exit(1)

    # Use sample for faster training (set to None for full dataset)
    # Full dataset: 6.3M transactions -> ~10M-node graph, blows past 16 GB VRAM.
    # Even 500k → ~880k nodes still OOMs with heads=4 in gat1 — reducing both:
    # 300k transactions ≈ 540k nodes, plus gat1 heads cut from 4→2, fits in 16GB.
    sample_size = 300_000

    data, scaler, feature_names = load_and_build_graph(sample_size=sample_size)

    print("\n" + "=" * 60)
    print("Building HybridGNN Model")
    print("=" * 60)

    in_channels = data.x.shape[1]
    model = HybridGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=2,
        dropout=0.3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HybridGNN")
    print(f"Input features: {in_channels}")
    print(f"Total parameters: {total_params:,}")

    model, best_val_auc = train_model(model, data, epochs=150, lr=0.003, patience=30)
    metrics = evaluate_model(model, data)

    if "error" not in metrics:
        save_artifacts(model, scaler, feature_names, metrics)

    print("\n" + "=" * 70)
    print("GNN V4 PAYSIM TRAINING COMPLETE")
    print("=" * 70)
    if "error" not in metrics:
        print(f"""
Final Performance:
  AUC-ROC: {metrics['auc_roc']:.4f}
  AUC-PR:  {metrics['auc_pr']:.4f}
  Best F1: {metrics['best_f1']:.4f}
""")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
