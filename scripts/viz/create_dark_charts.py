"""Regenerate all evaluation charts in dark mode + new feature-importance chart.

Sources:
  - outputs/evaluation/model_statistics.json   (confusion matrix, headline metrics)
  - outputs/evaluation/feature_importance.json (per-feature contribution)

If the test dataset and trained model are available, also re-runs inference to
regenerate the per-sample charts (ROC/PR curves, score distribution).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import dark_theme as dt

dt.apply()

EVAL_DIR = OUTPUTS_DIR / 'evaluation'


# ── Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix():
    stats = json.loads((EVAL_DIR / 'model_statistics.json').read_text())
    cm_data = stats['confusion_matrix']
    cm = np.array([[cm_data['true_negatives'],  cm_data['false_positives']],
                   [cm_data['false_negatives'], cm_data['true_positives']]])
    cm_pct = cm / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap=dt.HEATMAP_CMAP, aspect='equal')

    labels = ['Benign (0)', 'Criminal (1)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label',      fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix — Test Set (n={:,})'.format(stats['test_set_size']),
                 fontsize=15, fontweight='bold', pad=15)

    threshold = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=15, fontweight='bold',
                    color='white' if cm[i, j] >= threshold else dt.TEXT)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=dt.TEXT_DIM)
    cbar.outline.set_edgecolor(dt.EDGE)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=dt.TEXT_DIM)

    m = stats['metrics']
    metrics_line = (
        f"Acc {m['accuracy']*100:.2f}%  ·  F1 {m['f1_score']*100:.2f}%  ·  "
        f"Prec {m['precision']*100:.2f}%  ·  Rec {m['recall']*100:.2f}%  ·  "
        f"ROC-AUC {m['roc_auc']*100:.2f}%"
    )
    fig.text(0.5, 0.02, metrics_line, ha='center', fontsize=10,
             color=dt.TEXT_DIM, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = EVAL_DIR / 'confusion_matrix.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()
    print(f'  Saved: {out}')


# ── Feature importance (NEW chart) ──────────────────────────────────────────
def plot_feature_importance():
    data = json.loads((EVAL_DIR / 'feature_importance.json').read_text())
    fi = data['feature_importance']
    items = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    names  = [k.replace('_log', '') for k, _ in items]
    scores = np.array([v for _, v in items])

    fig, ax = plt.subplots(figsize=(11, 7))

    # Color bars by rank — top features warmer, tail features cooler
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0.85, 0.25, len(scores)))

    y_pos = np.arange(len(scores))[::-1]   # largest at the top
    bars = ax.barh(y_pos, scores * 100, color=colors, edgecolor=dt.EDGE, linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Importance (%)', fontsize=13, fontweight='bold')
    ax.set_title('Feature Importance — OptimalBitcoinGNN',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(scores) * 100 * 1.18)
    ax.grid(axis='x', color=dt.GRID, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(dt.EDGE)

    # Value labels at end of each bar
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 100 * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{score*100:.1f}%',
                va='center', fontsize=10, color=dt.TEXT, fontweight='bold')

    # Footnote
    top2 = scores[0] + scores[1]
    fig.text(0.5, 0.02,
             f"Top 2 features ({names[0]} + {names[1]}) account for {top2*100:.1f}% of total importance",
             ha='center', fontsize=10, color=dt.TEXT_DIM, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = EVAL_DIR / 'feature_importance.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()
    print(f'  Saved: {out}')


# ── ROC / PR / Score-distribution (need raw inference outputs) ──────────────
def maybe_run_inference():
    """Try to load the model and re-run inference. Return (labels, probs) or None."""
    try:
        import torch
        from src.graph.dataloader import get_test_loader
        from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
        from src.models.optimal_gnn import OptimalBitcoinGNN
        from src.models.utils import get_center_labels
    except Exception as e:
        print(f'  (skipping ROC/PR + score distribution: import failed: {e})')
        return None

    model_path = OUTPUTS_DIR / 'gnn_model.pt'
    if not model_path.exists():
        print('  (skipping ROC/PR + score distribution: gnn_model.pt not found)')
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimalBitcoinGNN(num_node_features=NUM_NODE_FEATURES,
                              num_edge_features=NUM_EDGE_FEATURES,
                              hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    temperature = 1.0
    temp_path = OUTPUTS_DIR / 'temperature.pt'
    if temp_path.exists():
        td = torch.load(temp_path, map_location=device, weights_only=True)
        temperature = float(td['temperature'])

    try:
        loader = get_test_loader(batch_size=64)
    except Exception as e:
        print(f'  (skipping ROC/PR + score distribution: test loader unavailable: {e})')
        return None

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) / temperature
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels = get_center_labels(batch)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_roc_and_pr(labels, probs):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    fpr, tpr, _    = roc_curve(labels, probs)
    prec, rec, _   = precision_recall_curve(labels, probs)
    roc_auc        = roc_auc_score(labels, probs)
    pr_auc         = average_precision_score(labels, probs)
    baseline       = float((labels == 1).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color=dt.ACCENT_BLUE, linewidth=2.5,
            label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color=dt.ACCENT_BLUE)
    ax.plot([0, 1], [0, 1], color=dt.TEXT_DIM, linestyle='--', linewidth=1, alpha=0.6,
            label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color=dt.GRID, alpha=0.4)

    # PR
    ax = axes[1]
    ax.plot(rec, prec, color=dt.CRIMINAL, linewidth=2.5,
            label=f'PR (AUC = {pr_auc:.4f})')
    ax.fill_between(rec, prec, alpha=0.15, color=dt.CRIMINAL)
    ax.axhline(y=baseline, color=dt.TEXT_DIM, linestyle='--', linewidth=1, alpha=0.6,
               label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall',    fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color=dt.GRID, alpha=0.4)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(dt.EDGE)

    plt.tight_layout()
    out = EVAL_DIR / 'roc_pr_curves.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()
    print(f'  Saved: {out}')


def plot_score_distribution(labels, probs):
    benign  = probs[labels == 0]
    crim    = probs[labels == 1]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5),
                             gridspec_kw={'height_ratios': [3, 1.5]})

    # Top: histograms
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(benign, bins=bins, alpha=0.75, color=dt.BENIGN,
            label=f'Benign (n={len(benign):,})', edgecolor=dt.BG, linewidth=0.6)
    ax.hist(crim,   bins=bins, alpha=0.75, color=dt.CRIMINAL,
            label=f'Criminal (n={len(crim):,})', edgecolor=dt.BG, linewidth=0.6)
    ax.axvline(x=0.5, color=dt.ACCENT_AMBER, linestyle='--', linewidth=1.6,
               label='Decision threshold (0.5)')
    ax.set_xlabel('Criminal Probability Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Criminal Probability Scores on Test Set',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.set_xlim(0, 1)
    ax.grid(axis='y', color=dt.GRID, alpha=0.4)

    # Bottom: jittered scatter
    ax2 = axes[1]
    rng = np.random.default_rng(42)

    def sample(arr, n=2000):
        if len(arr) > n:
            idx = rng.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

    pb = sample(benign); pc = sample(crim)
    jb = rng.uniform(-0.18, 0.18, size=len(pb))
    jc = rng.uniform(-0.18, 0.18, size=len(pc))
    ax2.scatter(pb, 0 + jb, alpha=0.45, s=10, color=dt.BENIGN,   label='Benign')
    ax2.scatter(pc, 1 + jc, alpha=0.45, s=10, color=dt.CRIMINAL, label='Criminal')
    ax2.axvline(x=0.5, color=dt.ACCENT_AMBER, linestyle='--', linewidth=1.6)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(['Benign', 'Criminal'], fontsize=12)
    ax2.set_xlabel('Criminal Probability Score', fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_title('Per-Sample Score Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', color=dt.GRID, alpha=0.4)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(dt.EDGE)

    plt.tight_layout()
    out = EVAL_DIR / 'score_distribution.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()
    print(f'  Saved: {out}')


def main():
    print('Generating dark-mode charts...')
    plot_confusion_matrix()
    plot_feature_importance()

    result = maybe_run_inference()
    if result is not None:
        labels, probs = result
        plot_roc_and_pr(labels, probs)
        plot_score_distribution(labels, probs)


if __name__ == '__main__':
    main()
