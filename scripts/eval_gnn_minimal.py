"""Minimal GNN inference on the current test intersection.

Bypasses PyG DataLoader iteration and the standard `--evaluate` pipeline
(which has been hanging on read() — suspected libomp/sklearn-mixing
contention on macOS). Strategy:

1. Pre-read every .pt file with plain torch.load (we know this is ~24ms each).
2. Hand-collate them into PyG Batches and run model.forward.
3. Compute metrics with numpy/torch only (no sklearn imports).

Writes the GNN metrics block back into outputs/evaluation/evaluation_results.json
and prints a summary.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph.config import (
    GRAPHS_DIR,
    NUM_EDGE_FEATURES,
    NUM_NODE_FEATURES,
    TEST_DATASET_PATH,
)
from src.models.optimal_gnn import OptimalBitcoinGNN


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # ROC-AUC by Mann-Whitney
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    # rank-based AUC
    scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(counts, dtype=np.float64)
    for i, r in zip(inv, ranks):
        sum_ranks[i] += r
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    sum_pos_ranks = ranks[:n_pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else 0.0
    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "roc_auc": float(auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n_samples": int(len(y_true)),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_csv = pd.read_csv(TEST_DATASET_PATH)
    test_pt_dir = os.path.join(GRAPHS_DIR, "test")
    test_pt_files = {f[:-3] for f in os.listdir(test_pt_dir) if f.endswith(".pt")}
    addrs = [a for a in test_csv["address"].astype(str).tolist() if a in test_pt_files]
    print(f"Test intersection: {len(addrs):,} addresses")

    print("Pre-loading .pt graphs...")
    t0 = time.time()
    graphs = []
    for i, addr in enumerate(addrs):
        path = os.path.join(test_pt_dir, f"{addr}.pt")
        graphs.append(torch.load(path, weights_only=False))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(addrs):,} loaded ({time.time() - t0:.1f}s elapsed)")
    print(f"All graphs loaded in {time.time() - t0:.1f}s")

    print("Loading OptimalBitcoinGNN...")
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    model_path = os.path.join(PROJECT_ROOT, "outputs", "gnn_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {model_path}")

    # Hand-batch and run forward
    print("Running inference...")
    t0 = time.time()
    BATCH = 64
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for start in range(0, len(graphs), BATCH):
            batch = Batch.from_data_list(graphs[start:start + BATCH]).to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = torch.softmax(out, dim=1)
            preds.append(out.argmax(dim=1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
            # center label = first node of each graph in the batch
            # graphs are batched contiguously so y[0::num_nodes_per_graph] won't work directly;
            # instead pull each Data's first y element from the source graphs list
            for g in graphs[start:start + BATCH]:
                labels.append(int(g.y[0].item()))
    print(f"Inference done in {time.time() - t0:.1f}s")

    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    y_true = np.array(labels)
    metrics = _binary_metrics(y_true, y_pred, y_prob)

    # Write result block
    eval_path = os.path.join(PROJECT_ROOT, "outputs", "evaluation", "evaluation_results.json")
    payload = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            payload = json.load(f)
    payload["gnn"] = metrics
    payload["test_samples"] = metrics["n_samples"]
    payload["timestamp"] = datetime.now().isoformat()
    with open(eval_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Updated {eval_path}")

    print("\n--- GNN metrics ---")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k}: {metrics[k]:.4f}")
    print(f"  n: {metrics['n_samples']}, positive: {metrics['n_positive']}")


if __name__ == "__main__":
    main()
