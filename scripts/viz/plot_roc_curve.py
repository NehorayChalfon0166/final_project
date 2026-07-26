"""3-model ROC overlay on the shared test set (light mode, poster-ready).

Loads the three trained models and scores every wallet in the current shared
test set (test CSV rows that have a cached ego-graph), then overlays their ROC
curves with each AUC in the legend:

    * OptimalBitcoinGNN  (ours)   -- outputs/gnn_model.pt
    * BasicGCN          (graph baseline)  -- outputs/baseline/gcn_model.pt
    * XGBoost           (tabular baseline) -- retrained on the train intersection

No model is trained except XGBoost, which is re-fit on the same train wallets
the GNN saw (matching scripts/compare_three_models.py) so the comparison is
apples-to-apples. AUC is rank-based, so the (no-temperature) softmax scores here
yield exactly the roc_auc values stored in three_model_comparison.json.

Usage:
    python scripts/viz/plot_roc_curve.py
    python scripts/viz/plot_roc_curve.py --out roc_curve_light.png
"""
import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.viz import light_theme as theme
from src.baselines.gcn_baseline import BasicGCN
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


def roc_curve_np(y_true: np.ndarray, scores: np.ndarray):
    """Stepwise ROC + AUC (trapezoid) with numpy — avoids importing sklearn."""
    order = np.argsort(-scores, kind='mergesort')
    y = y_true[order].astype(np.float64)
    P = y.sum()
    N = len(y) - P
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tps / P])
    fpr = np.concatenate([[0.0], fps / N])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _score_graph_model(model, loader, device, is_optimal: bool):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if is_optimal:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            p = torch.softmax(out, dim=1)
            probs.append(p[:, 1].cpu().numpy())
            labels.append(get_center_labels(batch).cpu().numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(probs)


def _score_xgboost(test_addresses, train_addresses):
    """Re-fit XGBoost on the train intersection, score the test intersection."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH)
    train_df = train_df[train_df['address'].isin(train_addresses)].reset_index(drop=True)
    test_df = test_df[test_df['address'].isin(test_addresses)].reset_index(drop=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    y_train = train_df['label'].values
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    y_test = test_df['label'].values.astype(np.int64)

    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        eval_metric='logloss', verbosity=0, random_state=42,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_test, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=str, default='roc_curve_light.png')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch_geometric.loader import DataLoader

    test_dataset = EgoGraphDataset(split='test')
    train_dataset = EgoGraphDataset(split='train')
    test_addresses = set(test_dataset.addresses)
    train_addresses = set(train_dataset.addresses)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"device={device}  shared test n={len(test_dataset):,}")

    # OptimalBitcoinGNN (ours)
    print("scoring OptimalGNN...")
    opt = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    opt.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, 'outputs', 'gnn_model.pt'),
        map_location=device, weights_only=True))
    y_opt, p_opt = _score_graph_model(opt, loader, device, is_optimal=True)

    # BasicGCN (graph baseline)
    print("scoring BasicGCN...")
    gcn = BasicGCN(num_node_features=NUM_NODE_FEATURES, hidden_dim=64, dropout=0.3).to(device)
    gcn.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, 'outputs', 'baseline', 'gcn_model.pt'),
        map_location=device, weights_only=True))
    y_gcn, p_gcn = _score_graph_model(gcn, loader, device, is_optimal=False)

    # XGBoost (tabular baseline)
    print("scoring XGBoost...")
    y_xgb, p_xgb = _score_xgboost(test_addresses, train_addresses)

    # (display label, color, linewidth, (y_true, prob))
    series = [
        ('OptimalGNN', theme.ACCENT_BLUE,   3.0, (y_opt, p_opt)),
        ('XGBoost',    theme.ACCENT_GREEN,  2.2, (y_xgb, p_xgb)),
        ('BasicGCN',   theme.ACCENT_PURPLE, 2.2, (y_gcn, p_gcn)),
    ]
    curves = []
    for label, color, lw, (yt, pr) in series:
        fpr, tpr, auc = roc_curve_np(yt, pr)
        curves.append((label, color, lw, fpr, tpr, auc))
        print(f"  {label:<30s} AUC = {auc:.4f}")
    # Sort legend best -> worst.
    curves.sort(key=lambda c: -c[5])

    import matplotlib.pyplot as plt
    theme.apply()
    fig, ax = plt.subplots(figsize=(7.2, 6.8))

    for label, color, lw, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, color=color, lw=lw, zorder=3,
                label=f'{label} = {auc:.3f}')
    # Light fill under the best (first) curve to draw the eye to it.
    best = curves[0]
    ax.fill_between(best[3], best[4], color=best[1], alpha=0.08, zorder=2)
    # Random diagonal kept for reference but excluded from the legend.
    ax.plot([0, 1], [0, 1], color=theme.TEXT_DIM, ls='--', lw=1.2, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC — criminal-wallet detection', fontsize=13.5, pad=10)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=10.5)
    ax.set_aspect('equal', adjustable='box')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'poster')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
