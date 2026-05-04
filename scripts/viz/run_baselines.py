"""Run XGBoost (tabular) and a basic 2-layer GCN baseline on the SAME train/test
split used by the OptimalBitcoinGNN, then compute the full metric suite for each.

Outputs:
  outputs/baselines/baseline_results.json
  outputs/baselines/comparison_metrics.png   (bar chart of F1/ROC/Brier/ECE)
  outputs/baselines/confusion_matrices.png   (3-way: OptimalGNN / GCN / XGBoost)
  outputs/baselines/roc_pr_comparison.png    (curves overlaid)
  outputs/baselines/reliability_comparison.png
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import dark_theme as dt
from src.graph.dataloader import get_train_val_loaders, get_test_loader
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels

dt.apply()

OUT_DIR = OUTPUTS_DIR / 'baselines'
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic 2-layer GCN baseline
# ────────────────────────────────────────────────────────────────────────────
class BasicGCN(nn.Module):
    """Textbook 2-layer GCN with mean-pool readout. No edge features, no
    attention, no residuals, no ghost handling — minimal graph baseline."""

    def __init__(self, in_dim=12, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.classifier(h)


# ────────────────────────────────────────────────────────────────────────────
# 2. Helpers
# ────────────────────────────────────────────────────────────────────────────
def extract_tabular(loader):
    """Return (X, y) — center-node features and labels — for tabular models."""
    Xs, ys = [], []
    for batch in loader:
        labels = get_center_labels(batch).cpu().numpy()
        # Center node = first node of each graph in batch (index via ptr)
        counts = torch.bincount(batch.batch)
        ptr = torch.zeros(counts.size(0) + 1, dtype=torch.long)
        torch.cumsum(counts, dim=0, out=ptr[1:])
        center_x = batch.x[ptr[:-1]].cpu().numpy()
        Xs.append(center_x)
        ys.append(labels)
    return np.concatenate(Xs), np.concatenate(ys)


def brier(labels, probs):
    return float(np.mean((probs - labels) ** 2))


def ece(labels, probs, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    n = len(probs)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        err += (mask.sum() / n) * abs(labels[mask].mean() - probs[mask].mean())
    return float(err)


def compute_all_metrics(labels, preds, probs):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds)),
        'recall': float(recall_score(labels, preds)),
        'f1': float(f1_score(labels, preds)),
        'roc_auc': float(roc_auc_score(labels, probs)),
        'pr_auc': float(average_precision_score(labels, probs)),
        'brier': brier(labels, probs),
        'ece': ece(labels, probs),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'n_samples': int(len(labels)),
    }


# ────────────────────────────────────────────────────────────────────────────
# 3. Train / eval routines
# ────────────────────────────────────────────────────────────────────────────
def train_gcn(num_epochs=25, batch_size=64, lr=1e-3):
    print('\n[GCN] Loading data...')
    train_loader, val_loader = get_train_val_loaders(batch_size=batch_size, num_workers=0)
    test_loader = get_test_loader(batch_size=batch_size, num_workers=0)

    model = BasicGCN(in_dim=NUM_NODE_FEATURES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    print(f'[GCN] Training {num_epochs} epochs on {DEVICE}...')
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, get_center_labels(batch))
            loss.backward()
            optimizer.step()
            total += loss.item()

        # Validation
        model.eval()
        vl, vp = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                vp.extend(out.argmax(dim=1).cpu().numpy())
                vl.extend(get_center_labels(batch).cpu().numpy())
        from sklearn.metrics import f1_score
        val_f1 = f1_score(vl, vp)
        print(f'  Epoch {epoch+1:3d}  loss={total/len(train_loader):.4f}  val_F1={val_f1:.4f}')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f'[GCN] Best val F1: {best_val_f1:.4f}')

    # Test
    model.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = torch.softmax(out, dim=1)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
            labels.extend(get_center_labels(batch).cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def train_xgboost():
    from xgboost import XGBClassifier
    print('\n[XGBoost] Loading data...')
    train_loader, _ = get_train_val_loaders(batch_size=128, num_workers=0)
    test_loader = get_test_loader(batch_size=128, num_workers=0)

    print('[XGBoost] Extracting tabular features...')
    X_train, y_train = extract_tabular(train_loader)
    X_test, y_test = extract_tabular(test_loader)
    print(f'  train: {X_train.shape}  test: {X_test.shape}')

    print('[XGBoost] Training...')
    model = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f'[XGBoost] Trained in {time.time()-t0:.1f}s')

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return y_test, preds, probs


def evaluate_optimal():
    print('\n[OptimalGNN] Loading model + running test inference...')
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES, hidden_dim=64,
    ).to(DEVICE)
    model.load_state_dict(torch.load(OUTPUTS_DIR / 'gnn_model.pt', map_location=DEVICE, weights_only=True))
    model.eval()

    T = 1.0
    tp = OUTPUTS_DIR / 'temperature.pt'
    if tp.exists():
        T = float(torch.load(tp, map_location=DEVICE, weights_only=True)['temperature'])

    test_loader = get_test_loader(batch_size=64, num_workers=0)
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) / T
            p = torch.softmax(out, dim=1)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
            labels.extend(get_center_labels(batch).cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


# ────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ────────────────────────────────────────────────────────────────────────────
def plot_metric_bars(results, save_path):
    """Side-by-side bars for key metrics."""
    keys = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
    pretty = ['Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
    models = list(results.keys())
    colors = {
        'OptimalGNN': dt.ACCENT_BLUE,
        'BasicGCN':   dt.ACCENT_PURPLE,
        'XGBoost':    dt.CRIMINAL,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Discrimination bars
    ax = axes[0]
    x = np.arange(len(keys))
    w = 0.26
    for i, m in enumerate(models):
        vals = [results[m]['metrics'][k] for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], edgecolor=dt.BG)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=9, color=dt.TEXT)
    ax.set_xticks(x); ax.set_xticklabels(pretty)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel('Score'); ax.set_title('Discrimination metrics', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(axis='y', color=dt.GRID, alpha=0.4)

    # Calibration bars
    ax = axes[1]
    keys_c = ['brier', 'ece']
    pretty_c = ['Brier (lower better)', 'ECE (lower better)']
    x = np.arange(len(keys_c))
    for i, m in enumerate(models):
        vals = [results[m]['metrics'][k] for k in keys_c]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], edgecolor=dt.BG)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.005, f'{v:.4f}', ha='center', fontsize=9, color=dt.TEXT)
    ax.set_xticks(x); ax.set_xticklabels(pretty_c)
    ax.set_ylabel('Error'); ax.set_title('Calibration metrics', fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(axis='y', color=dt.GRID, alpha=0.4)

    for ax in axes:
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax.spines[sp].set_color(dt.EDGE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()


def plot_confusion_matrices(results, save_path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5))
    if len(results) == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = r['metrics']['confusion_matrix']
        arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        pct = arr / arr.sum() * 100
        im = ax.imshow(arr, cmap=dt.HEATMAP_CMAP, aspect='equal')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign', 'Criminal'])
        ax.set_yticklabels(['Benign', 'Criminal'])
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        m = r['metrics']
        ax.set_title(f"{name}\nF1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}",
                     fontweight='bold')
        thresh = arr.max() * 0.55
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{arr[i,j]:,}\n({pct[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white' if arr[i, j] >= thresh else dt.TEXT)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()


def plot_roc_pr(results, save_path):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'OptimalGNN': dt.ACCENT_BLUE, 'BasicGCN': dt.ACCENT_PURPLE, 'XGBoost': dt.CRIMINAL}

    for name, r in results.items():
        labels = r['_labels']; probs = r['_probs']
        fpr, tpr, _ = roc_curve(labels, probs)
        prec, rec, _ = precision_recall_curve(labels, probs)
        axes[0].plot(fpr, tpr, color=colors[name], lw=2.2,
                     label=f'{name}  (AUC={r["metrics"]["roc_auc"]:.4f})')
        axes[1].plot(rec, prec, color=colors[name], lw=2.2,
                     label=f'{name}  (AUC={r["metrics"]["pr_auc"]:.4f})')

    axes[0].plot([0, 1], [0, 1], color=dt.TEXT_DIM, ls='--', lw=1, alpha=0.6, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0].set_title('ROC Curves', fontweight='bold')
    axes[0].legend(loc='lower right'); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].grid(color=dt.GRID, alpha=0.4)

    axes[1].set_xlabel('Recall', fontweight='bold')
    axes[1].set_ylabel('Precision', fontweight='bold')
    axes[1].set_title('Precision-Recall Curves', fontweight='bold')
    axes[1].legend(loc='lower left'); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].grid(color=dt.GRID, alpha=0.4)

    for ax in axes:
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax.spines[sp].set_color(dt.EDGE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()


def plot_reliability(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'OptimalGNN': dt.ACCENT_BLUE, 'BasicGCN': dt.ACCENT_PURPLE, 'XGBoost': dt.CRIMINAL}
    n_bins = 15
    edges = np.linspace(0, 1, n_bins + 1)
    for name, r in results.items():
        labels = r['_labels']; probs = r['_probs']
        idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
        confs, accs = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() < 5: continue
            confs.append(probs[m].mean()); accs.append(labels[m].mean())
        ece_v = r['metrics']['ece']
        ax.plot(confs, accs, marker='o', lw=2.2, color=colors[name], markersize=7,
                markeredgecolor=dt.BG, markeredgewidth=0.8,
                label=f'{name}  (ECE={ece_v:.4f})')
    ax.plot([0, 1], [0, 1], color=dt.TEXT_DIM, ls='--', lw=1, alpha=0.6, label='Perfect')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability', fontweight='bold')
    ax.set_ylabel('Empirical accuracy', fontweight='bold')
    ax.set_title('Reliability Diagram — All Models', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left'); ax.grid(color=dt.GRID, alpha=0.4)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(dt.EDGE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 5. Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    results = {}

    labels, preds, probs = evaluate_optimal()
    results['OptimalGNN'] = {'_labels': labels, '_probs': probs,
                             'metrics': compute_all_metrics(labels, preds, probs)}

    labels, preds, probs = train_gcn()
    results['BasicGCN'] = {'_labels': labels, '_probs': probs,
                           'metrics': compute_all_metrics(labels, preds, probs)}

    labels, preds, probs = train_xgboost()
    results['XGBoost'] = {'_labels': labels, '_probs': probs,
                          'metrics': compute_all_metrics(labels, preds, probs)}

    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)
    for name, r in results.items():
        m = r['metrics']
        print(f"\n{name}")
        print(f"  Acc {m['accuracy']*100:.2f}%  F1 {m['f1']*100:.2f}%  "
              f"ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}")
        print(f"  Brier {m['brier']:.4f}  ECE {m['ece']:.4f}  "
              f"FPR {m['fpr']*100:.2f}%  FNR {m['fnr']*100:.2f}%")

    # Save JSON (strip numpy arrays)
    out_json = {name: {'metrics': r['metrics']} for name, r in results.items()}
    (OUT_DIR / 'baseline_results.json').write_text(json.dumps(out_json, indent=2))
    print(f"\nSaved: {OUT_DIR / 'baseline_results.json'}")

    plot_metric_bars(results, OUT_DIR / 'comparison_metrics.png')
    plot_confusion_matrices(results, OUT_DIR / 'confusion_matrices.png')
    plot_roc_pr(results, OUT_DIR / 'roc_pr_comparison.png')
    plot_reliability(results, OUT_DIR / 'reliability_comparison.png')
    print(f'Saved: {OUT_DIR}/*.png')


if __name__ == '__main__':
    main()
