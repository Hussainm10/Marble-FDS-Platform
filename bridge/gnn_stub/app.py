"""GNN Graph Neural Network Fraud Engine — HybridGNN V4 Service.

Loads the v4 HybridGNN trained on PaySim mobile-money transaction graphs (vs v3
which was on Elliptic Bitcoin). v4 uses 42 input features and gat1 with 2 heads
(v3 used 173 features and 4 heads). The model class below is parameterized so
older checkpoints can also be loaded if you point ARTIFACTS at a v3 config —
the default is v4.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from torch_geometric.nn import GATConv, SAGEConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    GATConv = None
    SAGEConv = None

app = FastAPI(title="GNN Graph Fraud Engine", version="4.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gnn")

# --- Model Definitions ---

class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification (legacy v1)."""

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


class HybridGNN(nn.Module):
    """Hybrid GNN combining GAT and GraphSAGE with deep fusion.

    gat1_heads is parameterized so the same class can load both v4 (heads=2,
    used to fit a 540k-node graph in 16 GB VRAM) and legacy v3 (heads=4).
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.3, gat1_heads: int = 2):
        super().__init__()
        self.dropout = dropout

        # GAT branch - learns attention over neighbors. gat1_heads varies by checkpoint.
        self.gat1 = GATConv(in_channels, hidden_channels, heads=gat1_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * gat1_heads, hidden_channels, heads=2, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_channels * 2, hidden_channels, heads=1, dropout=dropout, concat=False)

        # SAGE branch - robust mean aggregation
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)

        # Normalization layers
        self.gat_norm1 = nn.LayerNorm(hidden_channels * gat1_heads)
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


# --- Model Loading ---

# In Docker: /app/artifacts, in local dev: ../gnn/artifacts
ARTIFACTS_DIR = Path("/app/artifacts") if Path("/app/artifacts").exists() else Path(__file__).parent.parent / "gnn" / "artifacts"

# Global model objects
model: Optional[HybridGNN] = None
scaler = None
config: dict = {}
best_threshold: float = 0.5
expected_in_channels: int = 42  # Updated when a model is loaded.
loaded_model_version: str = ""


def load_models():
    """Load all model artifacts at startup.

    Prefers v4 (PaySim) artifacts; falls back to v3 (Elliptic) if v4 isn't
    present. The architecture (gat1_heads) is inferred from the saved
    state_dict so v3 and v4 checkpoints both load cleanly.
    """
    global model, scaler, config, best_threshold, expected_in_channels, loaded_model_version

    if not HAS_TORCH_GEOMETRIC:
        logger.warning("torch_geometric not installed, running in stub mode")
        return False

    candidates = [
        ("v4", ARTIFACTS_DIR / "hybrid_v4_paysim.pt",
         ARTIFACTS_DIR / "scaler_hybrid_v4.pkl",
         ARTIFACTS_DIR / "config_hybrid_v4.json"),
        ("v3", ARTIFACTS_DIR / "hybrid_v3.pt",
         ARTIFACTS_DIR / "scaler_hybrid_v3.pkl",
         ARTIFACTS_DIR / "config_hybrid_v3.json"),
    ]

    for version, model_path, scaler_path, config_path in candidates:
        if not model_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)

            arch = config.get("architecture", {})
            hidden_channels = arch.get("hidden_channels", 128)
            dropout = arch.get("dropout", 0.3)
            in_channels = arch.get("in_channels", 42 if version == "v4" else 173)
            out_channels = arch.get("out_channels", 2)

            # Infer gat1_heads from the saved state — gat1 weight has shape
            # [in_channels, hidden_channels * heads]. v4 trained with heads=2,
            # v3 with heads=4. This means the same class loads both cleanly.
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            gat1_lin_w = state_dict.get("gat1.lin.weight", state_dict.get("gat1.lin_src.weight"))
            if gat1_lin_w is not None:
                gat1_heads = gat1_lin_w.shape[0] // hidden_channels
            else:
                gat1_heads = 2 if version == "v4" else 4

            model = HybridGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                dropout=dropout,
                gat1_heads=gat1_heads,
            )
            model.load_state_dict(state_dict)
            model.eval()

            scaler = joblib.load(scaler_path)
            best_threshold = config.get("best_threshold",
                                        config.get("metrics", {}).get("best_threshold", 0.5))
            expected_in_channels = in_channels
            loaded_model_version = version

            logger.info(
                "Loaded HybridGNN %s: in_channels=%d, hidden=%d, gat1_heads=%d, threshold=%.2f",
                version, in_channels, hidden_channels, gat1_heads, best_threshold,
            )
            return True

        except Exception as e:
            logger.error("Failed to load %s artifacts: %s", version, e)
            import traceback
            traceback.print_exc()
            continue

    logger.warning("No GNN checkpoints found, falling back to stub mode")
    return False


# --- Request / Response models ---

class NodeFeatures(BaseModel):
    """Features for a single node (account) in the transaction graph.

    For v4 (PaySim), the feature vector is 42-dimensional: send/recv amount
    stats, balance stats, step stats, transaction-type one-hots, and graph
    degree features. The exact ordered list is in
    `config_hybrid_v4.json::feature_names`.
    """
    features: list[float] = Field(default_factory=list, description="Feature vector matching the loaded model's in_channels (42 for v4)")
    neighbors: list[list[float]] = Field(default_factory=list, description="Neighbor node features (same dimensionality)")


class EvaluateRequest(BaseModel):
    decision_id: str = ""
    checkmarble_score: float = 0
    marbel_score: float = 0
    trigger_object_type: str = ""
    trigger_object_id: str = ""
    entity_id: str = ""
    # Optional: node features for real GNN scoring
    node_data: Optional[NodeFeatures] = None


class EntityGraphSignals(BaseModel):
    shared_devices: int = 0
    linked_wallets: int = 0
    self_circulation_chain: bool = False
    entity_cluster_size: int = 1


class EvaluateResponse(BaseModel):
    enhanced_score: float
    fraud_probability: float = 0.0
    threshold_used: float = 0.5
    is_fraud_prediction: bool = False
    entity_graph_signals: EntityGraphSignals = Field(default_factory=EntityGraphSignals)
    risk_factors: dict = Field(default_factory=dict)
    inference_mode: str = "stub"


# --- Stub Scoring (fallback) ---

def _deterministic_hash(s: str, mod: int) -> int:
    """Generate a deterministic pseudo-random number from a string."""
    h = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    return h % mod


def compute_stub_score(req: EvaluateRequest) -> EvaluateResponse:
    """Simulate GNN graph-based fraud scoring (stub mode)."""
    base = (req.checkmarble_score * 0.35) + (req.marbel_score * 0.65)
    entity = req.entity_id or req.trigger_object_id or req.decision_id
    boost = _deterministic_hash(entity, 15)
    enhanced = min(100.0, round(base + boost, 1))

    shared_devices = _deterministic_hash(f"{entity}_devices", 6)
    linked_wallets = _deterministic_hash(f"{entity}_wallets", 15)
    cluster_size = max(1, linked_wallets // 2)
    self_circ = _deterministic_hash(f"{entity}_circ", 4) == 0

    signals = EntityGraphSignals(
        shared_devices=shared_devices,
        linked_wallets=linked_wallets,
        self_circulation_chain=self_circ,
        entity_cluster_size=cluster_size,
    )

    risk_factors = {}
    if self_circ:
        risk_factors["circular_flow"] = round(0.20 + boost * 0.01, 2)
    if linked_wallets > 5:
        risk_factors["multi_wallet_access"] = round(0.10 + linked_wallets * 0.015, 2)
    if shared_devices > 2:
        risk_factors["device_sharing"] = round(0.08 + shared_devices * 0.04, 2)

    return EvaluateResponse(
        enhanced_score=enhanced,
        fraud_probability=enhanced / 100.0,
        threshold_used=0.5,
        is_fraud_prediction=enhanced > 50,
        entity_graph_signals=signals,
        risk_factors=risk_factors,
        inference_mode="stub",
    )


def compute_real_score(req: EvaluateRequest) -> EvaluateResponse:
    """Real GNN scoring using trained HybridGNN v3 model."""
    node_data = req.node_data

    if node_data is None or len(node_data.features) != expected_in_channels:
        # Wrong shape (or no features) — fall back to stub mode.
        result = compute_stub_score(req)
        result.inference_mode = "stub_no_features"
        return result

    # Prepare node features tensor
    x = torch.tensor([node_data.features], dtype=torch.float32)

    # Scale features
    x_scaled = torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32)

    # For single-node inference without graph structure, we create a minimal graph
    # In production, you'd want to pass actual neighbor connections
    if node_data.neighbors:
        # Build mini-batch graph with node and neighbors
        all_features = [node_data.features] + node_data.neighbors
        x_batch = torch.tensor(all_features, dtype=torch.float32)
        x_batch = torch.tensor(scaler.transform(x_batch.numpy()), dtype=torch.float32)

        # Create edges from node 0 to all neighbors
        num_neighbors = len(node_data.neighbors)
        src = [0] * num_neighbors + list(range(1, num_neighbors + 1))
        dst = list(range(1, num_neighbors + 1)) + [0] * num_neighbors
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # Single node with self-loop
        x_batch = x_scaled
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # Run inference
    with torch.no_grad():
        logits = model(x_batch, edge_index)
        probs = F.softmax(logits, dim=1)
        fraud_prob = probs[0, 1].item()  # Probability of class 1 (fraud)

    # Convert to 0-100 score (ensure Python float, not numpy)
    enhanced_score = float(round(fraud_prob * 100, 1))
    is_fraud = bool(fraud_prob >= best_threshold)

    # Generate graph signals (would need actual graph data in production)
    entity = req.entity_id or req.trigger_object_id or req.decision_id
    signals = EntityGraphSignals(
        shared_devices=len(node_data.neighbors) if node_data.neighbors else 0,
        linked_wallets=len(node_data.neighbors) if node_data.neighbors else 0,
        self_circulation_chain=False,
        entity_cluster_size=1 + len(node_data.neighbors) if node_data.neighbors else 1,
    )

    # Risk factors based on probability
    risk_factors = {}
    if fraud_prob > 0.7:
        risk_factors["high_fraud_probability"] = round(fraud_prob, 3)
    if len(node_data.neighbors) > 3:
        risk_factors["suspicious_connections"] = round(0.1 + len(node_data.neighbors) * 0.02, 3)

    return EvaluateResponse(
        enhanced_score=enhanced_score,
        fraud_probability=float(round(fraud_prob, 4)),
        threshold_used=float(best_threshold),
        is_fraud_prediction=is_fraud,
        entity_graph_signals=signals,
        risk_factors=risk_factors,
        inference_mode="real",
    )


# --- Endpoints ---

@app.on_event("startup")
async def startup():
    """Load models on startup."""
    success = load_models()
    if success:
        logger.info("GNN service started with REAL HybridGNN V3 model")
    else:
        logger.warning("GNN service started in STUB mode (models not loaded)")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    logger.info(
        "GNN evaluate: decision=%s marbel_score=%s entity=%s has_node_data=%s",
        req.decision_id,
        req.marbel_score,
        req.entity_id,
        req.node_data is not None,
    )

    if model is not None and req.node_data is not None:
        result = compute_real_score(req)
    else:
        result = compute_stub_score(req)

    logger.info(
        "GNN result: enhanced_score=%s prob=%s mode=%s",
        result.enhanced_score,
        result.fraud_probability,
        result.inference_mode,
    )
    return result


@app.post("/score_node")
async def score_node(node: NodeFeatures):
    """Direct node scoring endpoint (bypasses integration)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(node.features) != expected_in_channels:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_in_channels} features (model {loaded_model_version or 'unknown'})",
        )

    x = torch.tensor([node.features], dtype=torch.float32)
    x_scaled = torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32)

    # Build graph with neighbors if provided
    if node.neighbors:
        all_features = [node.features] + node.neighbors
        x_batch = torch.tensor(all_features, dtype=torch.float32)
        x_batch = torch.tensor(scaler.transform(x_batch.numpy()), dtype=torch.float32)
        num_neighbors = len(node.neighbors)
        src = [0] * num_neighbors + list(range(1, num_neighbors + 1))
        dst = list(range(1, num_neighbors + 1)) + [0] * num_neighbors
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        x_batch = x_scaled
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x_batch, edge_index)
        probs = F.softmax(logits, dim=1)
        fraud_prob = probs[0, 1].item()

    return {
        "fraud_probability": float(round(fraud_prob, 4)),
        "risk_score": float(round(fraud_prob * 100, 1)),
        "is_fraud": bool(fraud_prob >= best_threshold),
        "threshold": float(best_threshold),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "gnn",
        "version": "4.0.0",
        "model_loaded": model is not None,
        "model_version": loaded_model_version or None,
        "in_channels": expected_in_channels if model is not None else None,
        "inference_mode": "real" if model is not None else "stub",
        "model_type": "HybridGNN" if model is not None else None,
        "torch_geometric_available": HAS_TORCH_GEOMETRIC,
    }


@app.get("/model_info")
async def model_info():
    """Return model metadata and metrics."""
    if not config:
        return {"error": "Model not loaded"}

    return {
        "version": config.get("version"),
        "trained_at": config.get("trained_at"),
        "architecture": config.get("architecture"),
        "metrics": config.get("metrics"),
        "best_threshold": best_threshold,
    }
