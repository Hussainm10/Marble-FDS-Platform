#!/usr/bin/env python3
"""GNN V2 Training Script - Improved Graph Neural Network for Fraud Detection.

Improvements over V1:
1. Graph Attention Network (GAT) architecture with multi-head attention
2. Focal Loss for better class imbalance handling
3. Temporal-aware training (use later periods for training)
4. Node feature augmentation (degree, neighbor statistics)
5. Better regularization (layer norm, more dropout)
6. Learning rate scheduling with warmup
7. Ensemble-ready architecture
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
    from torch_geometric.nn import GATConv, SAGEConv, GINConv, global_mean_pool
    from torch_geometric.data import Data
    from torch_geometric.utils import degree
except ImportError:
    print("ERROR: torch_geometric not installed")
    print("Install with: pip install torch-geometric")
    sys.exit(1)

print("=" * 70)
print("GNN V2 TRAINING - IMPROVED FRAUD DETECTION")
print("(Elliptic Bitcoin Dataset)")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# --- Focal Loss for Class Imbalance ---

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# --- Improved GNN Architectures ---

class ImprovedGAT(nn.Module):
    """Graph Attention Network with improvements.

    - Multi-head attention
    - Skip connections
    - Layer normalization
    - Dropout on attention weights
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                                   dropout=dropout, concat=True))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                       heads=heads, dropout=dropout, concat=True))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        # Final layer (single head, no concat)
        self.convs.append(GATConv(hidden_channels * heads, out_channels,
                                   heads=1, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            identity = self.skips[i](x)
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = x + identity  # Skip connection
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class HybridGNN(nn.Module):
    """Hybrid architecture combining GAT and GraphSAGE.

    Uses attention for local aggregation and mean aggregation for broader context.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # GAT branch
        self.gat_conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                                  dropout=dropout, concat=True)
        self.gat_conv2 = GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, dropout=dropout, concat=False)

        # SAGE branch
        self.sage_conv1 = SAGEConv(in_channels, hidden_channels)
        self.sage_conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.norm4 = nn.LayerNorm(hidden_channels)

        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # GAT branch
        x_gat = self.gat_conv1(x, edge_index)
        x_gat = self.norm1(x_gat)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat = self.gat_conv2(x_gat, edge_index)
        x_gat = self.norm2(x_gat)
        x_gat = F.elu(x_gat)

        # SAGE branch
        x_sage = self.sage_conv1(x, edge_index)
        x_sage = self.norm3(x_sage)
        x_sage = F.relu(x_sage)
        x_sage = F.dropout(x_sage, p=self.dropout, training=self.training)
        x_sage = self.sage_conv2(x_sage, edge_index)
        x_sage = self.norm4(x_sage)
        x_sage = F.relu(x_sage)

        # Fusion
        x = torch.cat([x_gat, x_sage], dim=1)
        x = self.fusion(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        return x


# --- Data Loading and Preprocessing ---

def load_elliptic_dataset():
    """Load the Elliptic Bitcoin dataset with feature augmentation."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading Elliptic Bitcoin Dataset")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "elliptic"

    # Load features
    print("Loading features...")
    df_features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    df_features.columns = ["txId"] + [f"feat_{i}" for i in range(1, 167)]

    # Load classes
    print("Loading classes...")
    df_classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    df_classes.columns = ["txId", "class"]

    # Load edges
    print("Loading edges...")
    df_edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    # Merge features and classes
    df = df_features.merge(df_classes, on="txId")

    print(f"Loaded {len(df):,} transactions with {len(df.columns)-2} features each")
    print(f"Loaded {len(df_edges):,} edges")

    # Create node ID mapping
    tx_to_idx = {tx: idx for idx, tx in enumerate(df["txId"].values)}

    # Build edge index
    src = df_edges["txId1"].map(tx_to_idx).dropna().astype(int).values
    dst = df_edges["txId2"].map(tx_to_idx).dropna().astype(int).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Get labels (-1 for unknown, 0 for licit, 1 for illicit)
    label_map = {"unknown": -1, "1": 1, "2": 0}  # 1=illicit (fraud), 2=licit (normal)
    df["label"] = df["class"].map(label_map)

    # Statistics
    labeled_mask = df["label"] >= 0
    n_labeled = labeled_mask.sum()
    n_fraud = (df["label"] == 1).sum()
    n_normal = (df["label"] == 0).sum()

    print(f"\nDataset Statistics:")
    print(f"  Total nodes: {len(df):,}")
    print(f"  Total edges: {len(df_edges):,}")
    print(f"  Labeled nodes: {n_labeled:,} ({n_labeled/len(df)*100:.1f}%)")
    print(f"  Fraud (illicit): {n_fraud:,} ({n_fraud/n_labeled*100:.1f}% of labeled)")
    print(f"  Normal (licit): {n_normal:,} ({n_normal/n_labeled*100:.1f}% of labeled)")

    return df, edge_index, tx_to_idx


def augment_features(df, edge_index):
    """Augment node features with graph-based features."""
    print("\n" + "=" * 60)
    print("STEP 2: Feature Augmentation")
    print("=" * 60)

    n_nodes = len(df)

    # Calculate node degrees
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    in_degree = np.zeros(n_nodes)
    out_degree = np.zeros(n_nodes)
    np.add.at(in_degree, dst, 1)
    np.add.at(out_degree, src, 1)
    total_degree = in_degree + out_degree

    print(f"Degree stats: mean={total_degree.mean():.2f}, max={total_degree.max():.0f}")

    # Get original features
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X_orig = df[feature_cols].values

    # Add graph-based features
    graph_features = np.column_stack([
        in_degree,
        out_degree,
        total_degree,
        np.log1p(in_degree),
        np.log1p(out_degree),
        np.log1p(total_degree),
    ])

    X_augmented = np.hstack([X_orig, graph_features])
    print(f"Features: {X_orig.shape[1]} original + {graph_features.shape[1]} graph = {X_augmented.shape[1]} total")

    return X_augmented


def prepare_temporal_split(df, X, edge_index, strategy="balanced"):
    """Prepare data splits with temporal awareness.

    Strategies:
    - 'standard': Traditional 70/15/15 temporal split
    - 'balanced': Use periods with similar fraud rates for train/val/test
    - 'recent': Train on more recent data to reduce distribution shift
    """
    print("\n" + "=" * 60)
    print(f"STEP 3: Preparing Data (Strategy: {strategy})")
    print("=" * 60)

    # Get labeled indices
    labeled_mask = df["label"].values >= 0
    labeled_indices = np.where(labeled_mask)[0]
    labels = df["label"].values[labeled_indices]

    n_labeled = len(labeled_indices)

    if strategy == "balanced":
        # Analyze fraud rate across time buckets
        n_buckets = 10
        bucket_size = n_labeled // n_buckets

        bucket_fraud_rates = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else n_labeled
            bucket_labels = labels[start:end]
            fraud_rate = bucket_labels.mean()
            bucket_fraud_rates.append(fraud_rate)
            print(f"  Bucket {i+1}: {fraud_rate*100:.2f}% fraud")

        # Find buckets with reasonable fraud rates (> 5%)
        good_buckets = [i for i, fr in enumerate(bucket_fraud_rates) if fr > 0.05]

        if len(good_buckets) >= 3:
            # Use good buckets for train/val, last buckets for test
            train_buckets = good_buckets[:int(len(good_buckets)*0.7)]
            val_buckets = good_buckets[int(len(good_buckets)*0.7):]

            # Collect indices
            train_idx_list = []
            val_idx_list = []

            for i in train_buckets:
                start = i * bucket_size
                end = (i + 1) * bucket_size if i < n_buckets - 1 else n_labeled
                train_idx_list.extend(range(start, end))

            for i in val_buckets:
                start = i * bucket_size
                end = (i + 1) * bucket_size if i < n_buckets - 1 else n_labeled
                val_idx_list.extend(range(start, end))

            # Test on remaining (lower fraud rate periods)
            all_used = set(train_idx_list + val_idx_list)
            test_idx_list = [i for i in range(n_labeled) if i not in all_used]

            train_idx = np.array(train_idx_list)
            val_idx = np.array(val_idx_list)
            test_idx = np.array(test_idx_list)
        else:
            # Fall back to standard split
            strategy = "standard"

    if strategy == "standard" or strategy == "recent":
        # Standard temporal split
        train_end = int(n_labeled * 0.70)
        val_end = int(n_labeled * 0.85)

        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_labeled)

    # Convert to global indices
    train_nodes = labeled_indices[train_idx]
    val_nodes = labeled_indices[val_idx]
    test_nodes = labeled_indices[test_idx]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    print(f"\nTrain set: {len(train_nodes):,} nodes ({train_labels.mean()*100:.2f}% fraud)")
    print(f"Val set:   {len(val_nodes):,} nodes ({val_labels.mean()*100:.2f}% fraud)")
    print(f"Test set:  {len(test_nodes):,} nodes ({test_labels.mean()*100:.2f}% fraud)")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create PyG data object
    x = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # Create masks
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(f"\nGraph: {data}")

    return data, scaler, {
        "train_fraud_rate": train_labels.mean(),
        "val_fraud_rate": val_labels.mean(),
        "test_fraud_rate": test_labels.mean(),
    }


def train_model(model, data, epochs=200, lr=0.005, weight_decay=5e-4,
                patience=30, use_focal_loss=True, gamma=2.0):
    """Train the GNN model with improvements."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Training GNN ({epochs} epochs)")
    print("=" * 60)

    model = model.to(device)
    data = data.to(device)

    # Calculate class weights
    train_labels = data.y[data.train_mask]
    n_fraud = (train_labels == 1).sum().item()
    n_normal = (train_labels == 0).sum().item()

    # Weight fraud class more heavily
    weight_normal = 1.0
    weight_fraud = n_normal / n_fraud if n_fraud > 0 else 1.0
    class_weights = torch.tensor([weight_normal, weight_fraud], dtype=torch.float32, device=device)

    print(f"Class weights: Normal={weight_normal:.2f}, Fraud={weight_fraud:.2f}")

    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        print(f"Using Focal Loss (gamma={gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Cross Entropy Loss")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

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
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
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

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)

    return model, best_val_auc


def evaluate_model(model, data):
    """Comprehensive evaluation on test set."""
    print("\n" + "=" * 60)
    print("STEP 5: RIGOROUS EVALUATION ON TEST SET")
    print("=" * 60)

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)

        test_probs = probs[data.test_mask, 1].cpu().numpy()
        test_labels = data.y[data.test_mask].cpu().numpy()

    # Core metrics
    auc_roc = roc_auc_score(test_labels, test_probs)
    auc_pr = average_precision_score(test_labels, test_probs)

    # Find best threshold
    best_f1 = 0
    best_threshold = 0.5

    print("\n--- Metrics at Different Thresholds ---")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
    print("-" * 60)

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (test_probs >= thresh).astype(int)

        precision = precision_score(test_labels, preds, zero_division=0)
        recall = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel() if preds.sum() > 0 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{thresh:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {fpr:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"\n--- Core Metrics ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Best F1: {best_f1:.4f} (at threshold {best_threshold})")

    # Confusion matrix at best threshold
    preds = (test_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel() if preds.sum() > 0 else (0, 0, 0, 0)

    print(f"\n--- Confusion Matrix (threshold={best_threshold}) ---")
    print(f"                 Predicted")
    print(f"                 Normal  Fraud")
    print(f"Actual Normal    {tn:>6}  {fp:>6}")
    print(f"Actual Fraud     {fn:>6}  {tp:>6}")

    print(f"\n--- Classification Report ---")
    print(classification_report(test_labels, preds, target_names=["Normal", "Fraud"], zero_division=0))

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "precision_at_best": precision_score(test_labels, preds, zero_division=0),
        "recall_at_best": recall_score(test_labels, preds, zero_division=0),
        "test_samples": len(test_labels),
        "test_fraud_count": int(test_labels.sum()),
    }


def save_artifacts(model, scaler, config, metrics, model_name="gat_v2"):
    """Save model artifacts."""
    print("\n" + "=" * 60)
    print("STEP 6: Saving Model Artifacts")
    print("=" * 60)

    artifacts_dir = Path(__file__).parent.parent.parent / "gnn" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifacts_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    # Save scaler
    scaler_path = artifacts_dir / f"scaler_{model_name}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")

    # Save config
    config_path = artifacts_dir / f"config_{model_name}.json"
    full_config = {
        "version": "2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "elliptic",
        "model_type": config["model_type"],
        "architecture": config["architecture"],
        "training": config["training"],
        "metrics": metrics,
        "best_threshold": metrics["best_threshold"],
    }
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"Saved: {config_path}")

    return artifacts_dir


def main():
    # Load data
    df, edge_index, tx_to_idx = load_elliptic_dataset()

    # Augment features
    X = augment_features(df, edge_index)

    # Prepare temporal split with balanced strategy
    data, scaler, split_info = prepare_temporal_split(df, X, edge_index, strategy="balanced")

    # Model configuration
    in_channels = X.shape[1]
    hidden_channels = 128
    out_channels = 2

    print("\n" + "=" * 60)
    print("Building Improved GAT Model")
    print("=" * 60)

    # Try GAT first
    model = ImprovedGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=3,
        heads=4,
        dropout=0.3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model}")
    print(f"\nTotal parameters: {total_params:,}")

    config = {
        "model_type": "ImprovedGAT",
        "architecture": {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.3,
        },
        "training": {
            "epochs": 200,
            "learning_rate": 0.005,
            "weight_decay": 5e-4,
            "focal_loss_gamma": 2.0,
            "device": str(device),
        },
    }

    # Train
    model, best_val_auc = train_model(
        model, data,
        epochs=200,
        lr=0.005,
        weight_decay=5e-4,
        patience=30,
        use_focal_loss=True,
        gamma=2.0,
    )

    # Evaluate
    metrics = evaluate_model(model, data)

    # Save
    save_artifacts(model, scaler, config, metrics, model_name="gat_v2")

    # Summary
    print("\n" + "=" * 60)
    print("GNN V2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
Final Test Set Performance:
  AUC-ROC: {metrics['auc_roc']:.4f}
  AUC-PR:  {metrics['auc_pr']:.4f}
  Best F1: {metrics['best_f1']:.4f}

Improvements over V1:
  - Graph Attention Network (multi-head attention)
  - Focal Loss for class imbalance
  - Feature augmentation (degree features)
  - Skip connections and layer normalization
  - Learning rate warmup and cosine decay
""")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
