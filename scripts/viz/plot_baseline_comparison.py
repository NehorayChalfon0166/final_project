"""Grouped bar chart: one group per metric, one bar per model (light mode).

Reads `outputs/evaluation/three_model_comparison.json` (XGBoost, BasicGCN,
OptimalGNN — all evaluated on the same shared test wallets). The x-axis is the
metric; within each metric there is one bar per model, and the bar height is the
metric value as a percentage. Model colors match the ROC overlay so the two
poster charts read as a set.

No model inference — pure plotting from stored results.

Usage:
    python scripts/viz/plot_baseline_comparison.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.viz import light_theme as theme

# Metric key in the JSON -> display label (x-axis groups).
METRICS = [
    ('accuracy',  'Accuracy'),
    ('precision', 'Precision'),
    ('recall',    'Recall'),
    ('f1',        'F1'),
    ('roc_auc',   'ROC-AUC'),
]

# Model key -> (legend label, bar color). Order = best -> baselines, left to
# right within each metric group. Colors match plot_roc_curve.py.
MODELS = [
    ('OptimalGNN', 'OptimalGNN (ours)',          theme.ACCENT_BLUE),
    ('XGBoost',    'XGBoost (tabular baseline)',  theme.ACCENT_GREEN),
    ('BasicGCN',   'BasicGCN (graph baseline)',   theme.ACCENT_PURPLE),
]


def main():
    theme.apply()

    results_path = os.path.join(PROJECT_ROOT, 'outputs', 'evaluation',
                                'three_model_comparison.json')
    with open(results_path) as f:
        data = json.load(f)
    models = data['models']
    n_test = data.get('shared_test_size')
    n_train = data.get('shared_train_size')

    x = np.arange(len(METRICS))
    n_models = len(MODELS)
    bar_w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 6.2))

    for i, (mkey, mlabel, color) in enumerate(MODELS):
        vals = [models[mkey][metric] * 100 for metric, _ in METRICS]
        offsets = x + (i - (n_models - 1) / 2) * bar_w
        # Make "ours" pop with a thin dark edge; baselines stay flat.
        edge = theme.BENIGN_DARK if mkey == 'OptimalGNN' else theme.BG
        is_ours = mkey == 'OptimalGNN'
        bars = ax.bar(offsets, vals, bar_w, label=mlabel, color=color,
                      edgecolor=edge, linewidth=1.0 if is_ours else 0.6,
                      zorder=3)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2,
                    f'{v:.1f}', ha='center', va='bottom',
                    fontsize=9 if is_ours else 7.5,
                    fontweight='bold' if is_ours else 'normal',
                    color=theme.TEXT if is_ours else theme.TEXT_DIM, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=12)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('Model performance across metrics', fontsize=14, pad=12)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=n_models,
              frameon=False, fontsize=10.5)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    fig.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'poster')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'baseline_comparison_light.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', transparent=True)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
