#!/usr/bin/env python3
"""Render the GNN v4 learning curve PNG from runs/gnn_v4_paysim/metrics_history.json.

Usable any time during or after training. Also prints a one-line summary of
the latest epoch and the best validation AUC so far.
"""

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent.parent / "runs" / "gnn_v4_paysim"
METRICS = RUN_DIR / "metrics_history.json"
CURVE = RUN_DIR / "learning_curve.png"


def main():
    if not METRICS.exists():
        print(f"No metrics yet at {METRICS}. Has training started?")
        sys.exit(1)

    with open(METRICS) as f:
        history = json.load(f)
    if not history:
        print("metrics_history.json is empty.")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.suptitle(f"GNN v4 — epoch {epochs[-1]} / loss {loss[-1]:.4f} / "
                 f"val AUC {val_auc[-1]:.4f} / best AUC {max(val_auc):.4f}")
    fig.tight_layout()
    fig.savefig(CURVE, dpi=110)
    plt.close(fig)

    last = history[-1]
    print(f"Wrote {CURVE}")
    print(f"Latest: epoch {last['epoch']} | loss {last['loss']:.4f} | "
          f"val_auc {last['val_auc']:.4f} | val_ap {last['val_ap']:.4f}")
    print(f"Best val AUC so far: {max(val_auc):.4f} (at epoch "
          f"{epochs[val_auc.index(max(val_auc))]})")


if __name__ == "__main__":
    main()
