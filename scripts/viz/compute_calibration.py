"""Compute Brier score, ECE, and generate a reliability diagram."""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

import dark_theme as dt
from src.graph.dataloader import get_test_loader
from src.graph.config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels

dt.apply()

EVAL_DIR = OUTPUTS_DIR / 'evaluation'


def run_inference(temperature=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    model.load_state_dict(
        torch.load(OUTPUTS_DIR / 'gnn_model.pt', map_location=device, weights_only=True)
    )
    model.eval()

    temp_path = OUTPUTS_DIR / 'temperature.pt'
    if temp_path.exists():
        td = torch.load(temp_path, map_location=device, weights_only=True)
        temperature = float(td['temperature'])

    labels, probs_raw, probs_cal = [], [], []
    loader = get_test_loader(batch_size=64)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs_raw.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            probs_cal.append(torch.softmax(logits / temperature, dim=1)[:, 1].cpu().numpy())
            labels.append(get_center_labels(batch).cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(probs_raw),
        np.concatenate(probs_cal),
        temperature,
    )


def brier_score(labels, probs):
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(labels, probs, n_bins=15):
    """Standard ECE: bin by predicted prob, weight by bin size, sum |acc - conf|."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_data(labels, probs, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    centers, accs, confs, sizes = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        accs.append(float(labels[mask].mean()))
        confs.append(float(probs[mask].mean()))
        sizes.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(confs), np.array(sizes)


def plot_reliability(labels, probs_raw, probs_cal, temperature, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, probs, title, color in [
        (axes[0], probs_raw, 'Before T-scaling', dt.CRIMINAL),
        (axes[1], probs_cal, f'After T-scaling (T = {temperature:.3f})', dt.BENIGN),
    ]:
        centers, accs, confs, sizes = reliability_data(labels, probs)
        ece = expected_calibration_error(labels, probs)
        brier = brier_score(labels, probs)

        ax.plot([0, 1], [0, 1], color=dt.TEXT_DIM, linestyle='--', linewidth=1, alpha=0.6,
                label='Perfect calibration')
        ax.plot(confs, accs, color=color, linewidth=2.2, marker='o', markersize=7,
                markeredgecolor=dt.BG, markeredgewidth=1.0,
                label=f'ECE = {ece:.4f}\nBrier = {brier:.4f}')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Empirical accuracy',     fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(color=dt.GRID, alpha=0.4)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(dt.EDGE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()


def main():
    print('Running inference...')
    labels, probs_raw, probs_cal, T = run_inference()

    metrics = {
        'temperature': T,
        'before_calibration': {
            'brier_score': brier_score(labels, probs_raw),
            'ece':         expected_calibration_error(labels, probs_raw),
        },
        'after_calibration': {
            'brier_score': brier_score(labels, probs_cal),
            'ece':         expected_calibration_error(labels, probs_cal),
        },
    }

    print(json.dumps(metrics, indent=2))

    out_json = EVAL_DIR / 'calibration_metrics.json'
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f'Saved: {out_json}')

    out_png = EVAL_DIR / 'reliability_diagram.png'
    plot_reliability(labels, probs_raw, probs_cal, T, out_png)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
