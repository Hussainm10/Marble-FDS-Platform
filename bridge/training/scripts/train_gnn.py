"""Train GNN fraud detection model on Elliptic Bitcoin dataset.

This script:
1. Loads Elliptic Bitcoin transaction graph dataset
2. Preprocesses nodes, edges, and features
3. Trains a Graph Neural Network (GraphSAGE) for fraud detection
4. Evaluates on held-out test set
5. Exports model artifacts for serving

The Elliptic dataset contains:
- ~203k Bitcoin transactions (nodes)
- ~234k edges (transaction flows)
- 166 features per transaction
- Labels: 1=illicit, 2=licit, unknown=unlabeled

Usage:
    python train_gnn.py [--epochs 100] [--cpu]
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "elliptic"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "gnn" / "artifacts"

# Training config
RANDOM_STATE = 42
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4


def load_elliptic_data():
    """Load Elliptic Bitcoin dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading Elliptic Bitcoin Dataset")
    print("=" * 60)

    # Load features
    print("Loading features...")
    features_df = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    # First column is txId, rest are features (166 features)
    node_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 1:].values
    print(f"Loaded {len(node_ids):,} transactions with {features.shape[1]} features each")

    # Load classes
    print("Loading classes...")
    classes_df = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    # Map: 1=illicit (fraud), 2=licit (normal), unknown=unlabeled
    class_map = {"1": 1, "2": 0, "unknown": -1}  # 1=fraud, 0=normal, -1=unknown
    classes_df["label"] = classes_df["class"].map(class_map)

    # Create node_id to index mapping
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Get labels aligned with features
    labels = np.full(len(node_ids), -1)  # Default unknown
    for _, row in classes_df.iterrows():
        if row["txId"] in node_to_idx:
            labels[node_to_idx[row["txId"]]] = row["label"]

    # Load edges
    print("Loading edges...")
    edges_df = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

    # Filter edges to only include nodes in our feature set
    valid_edges = edges_df[
        edges_df["txId1"].isin(node_to_idx) & edges_df["txId2"].isin(node_to_idx)
    ]

    # Convert to edge index format
    edge_index = np.array([
        [node_to_idx[row["txId1"]], node_to_idx[row["txId2"]]]
        for _, row in valid_edges.iterrows()
    ]).T

    print(f"Loaded {edge_index.shape[1]:,} edges")

    # Statistics
    labeled_mask = labels != -1
    fraud_mask = labels == 1
    normal_mask = labels == 0

    print(f"\nDataset Statistics:")
    print(f"  Total nodes: {len(node_ids):,}")
    print(f"  Total edges: {edge_index.shape[1]:,}")
    print(f"  Labeled nodes: {labeled_mask.sum():,} ({labeled_mask.sum()/len(node_ids)*100:.1f}%)")
    print(f"  Fraud (illicit): {fraud_mask.sum():,} ({fraud_mask.sum()/labeled_mask.sum()*100:.1f}% of labeled)")
    print(f"  Normal (licit): {normal_mask.sum():,} ({normal_mask.sum()/labeled_mask.sum()*100:.1f}% of labeled)")

    return features, labels, edge_index, node_ids


def prepare_data(features, labels, edge_index):
    """Prepare data for GNN training with temporal split."""
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Data (Temporal Split)")
    print("=" * 60)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Get labeled indices
    labeled_idx = np.where(labels != -1)[0]
    n_labeled = len(labeled_idx)

    # Temporal split: first 70% train, next 15% val, last 15% test
    # This simulates real-world scenario where we train on past, predict future
    train_size = int(0.7 * n_labeled)
    val_size = int(0.15 * n_labeled)

    train_idx = labeled_idx[:train_size]
    val_idx = labeled_idx[train_size:train_size + val_size]
    test_idx = labeled_idx[train_size + val_size:]

    print(f"Train set: {len(train_idx):,} nodes")
    print(f"Val set:   {len(val_idx):,} nodes")
    print(f"Test set:  {len(test_idx):,} nodes")

    # Check class distribution
    print(f"\nTrain fraud rate: {labels[train_idx].sum()/len(train_idx)*100:.2f}%")
    print(f"Val fraud rate:   {labels[val_idx].sum()/len(val_idx)*100:.2f}%")
    print(f"Test fraud rate:  {labels[test_idx].sum()/len(test_idx)*100:.2f}%")

    return features_scaled, train_idx, val_idx, test_idx, scaler


def build_model(num_features, hidden_channels, num_layers, dropout):
    """Build GraphSAGE model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GraphSAGE(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Output layer
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

    model = GraphSAGE(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=2,  # Binary: fraud/normal
        num_layers=num_layers,
        dropout=dropout
    )

    return model


def train_gnn(model, data, train_idx, val_idx, labels, epochs, device):
    """Train the GNN model."""
    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print(f"STEP 3: Training GNN ({epochs} epochs)")
    print("=" * 60)

    model = model.to(device).float()
    data.x = data.x.float()
    data = data.to(device)

    # Class weights for imbalanced data
    train_labels = labels[train_idx]
    n_fraud = (train_labels == 1).sum()
    n_normal = (train_labels == 0).sum()
    weight = torch.tensor([1.0, float(n_normal) / float(n_fraud)], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], labels_tensor[train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = F.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
            val_true = labels_tensor[val_mask].cpu().numpy()

            val_auc = roc_auc_score(val_true, val_pred)
            val_ap = average_precision_score(val_true, val_pred)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    return model


def evaluate_model(model, data, test_idx, labels, device):
    """Evaluate model on test set."""
    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("STEP 4: RIGOROUS EVALUATION ON TEST SET")
    print("=" * 60)

    model.eval()
    data.x = data.x.float()
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    test_mask[test_idx] = True

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred_proba = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        test_true = labels_tensor[test_mask].cpu().numpy()

    # Metrics at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n--- Metrics at Different Thresholds ---")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (test_pred_proba >= thresh).astype(int)
        prec = precision_score(test_true, y_pred, zero_division=0)
        rec = recall_score(test_true, y_pred, zero_division=0)
        f1 = f1_score(test_true, y_pred, zero_division=0)

        tn = sum((test_true == 0) & (y_pred == 0))
        fp = sum((test_true == 0) & (y_pred == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fpr:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Core metrics
    auc_roc = roc_auc_score(test_true, test_pred_proba)
    auc_pr = average_precision_score(test_true, test_pred_proba)

    print(f"\n--- Core Metrics ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Best F1: {best_f1:.4f} (at threshold {best_threshold})")

    # Confusion matrix
    y_pred_best = (test_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(test_true, y_pred_best)

    print(f"\n--- Confusion Matrix (threshold={best_threshold}) ---")
    print(f"                 Predicted")
    print(f"                 Normal  Fraud")
    print(f"Actual Normal    {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"Actual Fraud     {cm[1,0]:>6}  {cm[1,1]:>6}")

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    print(f"\n--- Business Metrics ---")
    print(f"True Positives (fraud caught):  {tp:,}")
    print(f"False Positives (false alarms): {fp:,}")
    print(f"False Negatives (fraud missed): {fn:,}")
    print(f"True Negatives (clean cleared): {tn:,}")

    print(f"\n--- Classification Report ---")
    print(classification_report(test_true, y_pred_best, target_names=["Normal", "Fraud"]))

    metrics = {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "precision_at_best": float(precision_score(test_true, y_pred_best, zero_division=0)),
        "recall_at_best": float(recall_score(test_true, y_pred_best, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
        "test_samples": int(len(test_true)),
        "test_fraud_count": int(sum(test_true)),
    }

    return metrics


def save_artifacts(model, scaler, metrics, args):
    """Save model artifacts."""
    import torch

    print("\n" + "=" * 60)
    print("STEP 5: Saving Model Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = ARTIFACTS_DIR / "graphsage_v1.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    # Save scaler
    import joblib
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")

    # Save config
    config = {
        "version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "elliptic",
        "model_type": "GraphSAGE",
        "architecture": {
            "hidden_channels": HIDDEN_CHANNELS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": "cpu" if args.cpu else "cuda",
        },
        "metrics": metrics,
        "best_threshold": metrics["best_threshold"],
    }

    config_path = ARTIFACTS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train GNN on Elliptic dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    print("=" * 60)
    print("GNN FRAUD DETECTION MODEL TRAINING")
    print("(Elliptic Bitcoin Dataset)")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check device
    import torch
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"\nUsing device: CPU")
        if not args.cpu and not torch.cuda.is_available():
            print("(CUDA not available, falling back to CPU)")
    else:
        device = torch.device("cuda")
        print(f"\nUsing device: CUDA ({torch.cuda.get_device_name(0)})")

    # Load data
    features, labels, edge_index, node_ids = load_elliptic_data()

    # Prepare data
    features_scaled, train_idx, val_idx, test_idx, scaler = prepare_data(features, labels, edge_index)

    # Convert to PyTorch Geometric Data object
    from torch_geometric.data import Data

    x = torch.tensor(features_scaled, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index_tensor)
    print(f"\nGraph: {data}")

    # Build model
    print("\n" + "=" * 60)
    print("Building GraphSAGE Model")
    print("=" * 60)

    model = build_model(
        num_features=features.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train
    model = train_gnn(model, data, train_idx, val_idx, labels, args.epochs, device)

    # Evaluate
    data = data.to(device)
    metrics = evaluate_model(model, data, test_idx, labels, device)

    # Save
    save_artifacts(model, scaler, metrics, args)

    print("\n" + "=" * 60)
    print("GNN TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Test Set Performance:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"  Best F1: {metrics['best_f1']:.4f}")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
