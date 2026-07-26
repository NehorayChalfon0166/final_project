"""Evaluate the saved BasicGCN checkpoint on the current test set.

Inference-only — no training, no XGBoost imports (avoids the libomp deadlock).
Reads outputs/baseline/gcn_model.pt and writes updated outputs/baseline/gcn_results.json.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baselines.gcn_baseline import BasicGCN
from src.graph.config import NUM_NODE_FEATURES
from src.graph.dataloader import get_test_loader
from src.models.utils import get_center_labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader = get_test_loader(batch_size=128, num_workers=0)
    print(f"Test graphs: {len(test_loader.dataset):,}")

    model = BasicGCN(num_node_features=NUM_NODE_FEATURES, hidden_dim=64, dropout=0.3).to(device)
    model_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "gcn_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {model_path}")

    preds, probs, labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            p = torch.softmax(out, dim=1)
            preds.append(out.argmax(dim=1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
            labels.append(get_center_labels(batch).cpu().numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
    }
    print(json.dumps(metrics, indent=2))

    out_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "gcn_results.json")
    with open(out_path) as f:
        payload = json.load(f)
    payload["results"] = {**metrics, "training_time": payload["results"].get("training_time")}
    payload["evaluated_at"] = datetime.now().isoformat()
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()
