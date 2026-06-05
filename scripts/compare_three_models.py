"""Evaluate XGBoost, BasicGCN, and OptimalBitcoinGNN on the same test addresses.

The XGBoost test set on disk (test_dataset.csv: 21,718 rows) is larger than the
GNN test set (only addresses with a cached ego-graph: 7,840 rows). To compare
apples-to-apples, this script restricts the XGBoost evaluation to the same
7,840 wallets that the two graph models see.

Outputs outputs/evaluation/three_model_comparison.json + prints a side-by-side
table.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baselines.gcn_baseline import BasicGCN
from src.baselines.xgboost_baseline import XGBoostBaseline
from src.graph.config import (
    FEATURE_COLUMNS,
    NUM_EDGE_FEATURES,
    NUM_NODE_FEATURES,
    TEST_DATASET_PATH,
    TRAIN_DATASET_PATH,
)
from src.graph.dataloader import EgoGraphDataset
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EVAL_DIR = os.path.join(OUTPUTS_DIR, "evaluation")


def _binary_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }


def _evaluate_graph_model(model, loader, device) -> dict:
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if isinstance(model, OptimalBitcoinGNN):
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            p = torch.softmax(out, dim=1)
            preds.append(out.argmax(dim=1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
            labels.append(get_center_labels(batch).cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return _binary_metrics(y_true, y_pred, y_prob)


def _evaluate_xgboost_on_subset(test_addresses: set, train_addresses: set) -> dict:
    """Fit XGBoost on the train_dataset.csv rows whose address has a cached
    .pt graph, evaluate on the same-protocol test subset. This matches the
    GNN's training set row-for-row instead of giving XGBoost ~39k extra
    training samples the GNN never saw."""
    baseline = XGBoostBaseline()
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH)
    train_df = train_df[train_df["address"].isin(train_addresses)].reset_index(drop=True)
    test_df = test_df[test_df["address"].isin(test_addresses)].reset_index(drop=True)
    print(f"  XGBoost: train={len(train_df):,}, test={len(test_df):,} (both filtered to graph addresses)")

    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    y_train = train_df["label"].values
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    y_test = test_df["label"].values

    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return _binary_metrics(y_test, y_pred, y_prob)


def _print_table(rows: dict, metric_order: list) -> None:
    width = 14
    headers = ["Metric"] + list(rows.keys())
    line = f"{'Metric':<{width}}" + "".join(f"{h:>{width}}" for h in rows.keys())
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for m in metric_order:
        cells = []
        for model_name in rows:
            v = rows[model_name].get(m)
            if v is None:
                cells.append("N/A")
            elif isinstance(v, float):
                if m == "roc_auc":
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(f"{v*100:.2f}%")
            else:
                cells.append(str(v))
        print(f"{m:<{width}}" + "".join(f"{c:>{width}}" for c in cells))
    print("=" * len(line))


def main() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build the shared train+test sets: cached ego-graphs for each split.
    test_dataset = EgoGraphDataset(split="test")
    test_addresses = set(test_dataset.addresses)
    train_dataset = EgoGraphDataset(split="train")
    train_addresses = set(train_dataset.addresses)
    print(f"Shared train set: {len(train_addresses):,} addresses with cached graphs")
    print(f"Shared test set: {len(test_addresses):,} addresses with cached graphs")

    loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 2. OptimalBitcoinGNN
    print("\n[1/3] Evaluating OptimalBitcoinGNN...")
    opt_model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    opt_path = os.path.join(OUTPUTS_DIR, "gnn_model.pt")
    opt_model.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
    opt_metrics = _evaluate_graph_model(opt_model, loader, device)

    # 3. BasicGCN
    print("\n[2/3] Evaluating BasicGCN...")
    gcn_model = BasicGCN(num_node_features=NUM_NODE_FEATURES, hidden_dim=64, dropout=0.3).to(device)
    gcn_path = os.path.join(OUTPUTS_DIR, "baseline", "gcn_model.pt")
    gcn_model.load_state_dict(torch.load(gcn_path, map_location=device, weights_only=True))
    gcn_metrics = _evaluate_graph_model(gcn_model, loader, device)

    # 4. XGBoost on same subset
    print("\n[3/3] Evaluating XGBoost on the same train+test subsets...")
    xgb_metrics = _evaluate_xgboost_on_subset(test_addresses, train_addresses)

    rows = {
        "XGBoost": xgb_metrics,
        "BasicGCN": gcn_metrics,
        "OptimalGNN": opt_metrics,
    }
    metric_order = ["accuracy", "precision", "recall", "f1", "roc_auc", "n_samples"]
    print("\n")
    _print_table(rows, metric_order)

    payload = {
        "shared_test_size": len(test_addresses),
        "models": rows,
        "notes": (
            "All three models are trained and evaluated on exactly the same "
            "wallets: the CSV rows whose address has a cached .pt ego-graph. "
            "XGBoost is retrained on the train intersection (not the full "
            "train_dataset.csv) so it sees the same training samples the GNN "
            "saw. Train/test sizes are reported per-model in n_samples."
        ),
        "shared_train_size": len(train_addresses),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(EVAL_DIR, "three_model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
