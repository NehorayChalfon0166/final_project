"""Plot the calibrated risk-score / confidence distribution for train/test.

Scores every cached ego-graph in `graph_data/graphs/{train,test}/` through the
trained `OptimalBitcoinGNN`, applies temperature scaling (T from
`outputs/temperature.pt`), and saves TWO standalone images per split:

    <base>_riskscore.png   : risk score P[criminal] histogram, split by true label
    <base>_confidence.png  : confidence |p - 0.5| * 2 histogram (all samples)

Defaults to the test split only. With multiple splits the split name is inserted
into each filename (<base>_<split>_riskscore.png).

Usage:
    python scripts/viz/plot_confidence_distribution.py                       # test only
    python scripts/viz/plot_confidence_distribution.py --theme light --out confidence_distribution_light.png
    python scripts/viz/plot_confidence_distribution.py --splits train test   # both
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.viz import dark_theme, light_theme
from src.graph.config import NUM_EDGE_FEATURES, NUM_NODE_FEATURES
from src.graph.dataloader import EgoGraphDataset
from src.models.optimal_gnn import OptimalBitcoinGNN
from src.models.utils import get_center_labels


def score_split(model, split: str, temperature: float, device, batch_size: int = 64):
    dataset = EgoGraphDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    risk_scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'score {split}', total=len(loader), mininterval=1.0):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = F.softmax(logits / temperature, dim=1)
            risk_scores.append(probs[:, 1].cpu().numpy().astype(np.float32))
            labels.append(get_center_labels(batch).cpu().numpy().astype(np.int8))
            del batch, logits, probs
        gc.collect()
    return np.concatenate(risk_scores), np.concatenate(labels)


def _summary(risk: np.ndarray, lbl: np.ndarray) -> dict:
    conf = np.abs(risk - 0.5) * 2
    return {
        'n': int(len(risk)),
        'n_benign': int((lbl == 0).sum()),
        'n_criminal': int((lbl == 1).sum()),
        'mean_confidence': float(conf.mean()),
        'median_confidence': float(np.median(conf)),
        'frac_high_conf_>0.8': float((conf > 0.8).mean()),
        'frac_low_conf_<0.2':  float((conf < 0.2).mean()),
    }


def _minimal(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _plot_risk(ax, name, risk, lbl, bins, theme):
    ax.hist(risk[lbl == 0], bins=bins, color=theme.BENIGN, alpha=0.75,
            label=f'benign  (n={int((lbl == 0).sum()):,})',
            edgecolor=theme.BENIGN_DARK, linewidth=0.4)
    ax.hist(risk[lbl == 1], bins=bins, color=theme.CRIMINAL, alpha=0.75,
            label=f'criminal (n={int((lbl == 1).sum()):,})',
            edgecolor=theme.CRIMINAL_DARK, linewidth=0.4)
    ax.axvline(0.5, color=theme.ACCENT_AMBER, ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('P[criminal] (temperature-scaled)')
    ax.set_ylabel('count')
    ax.legend(frameon=False, labelcolor=theme.TEXT)
    _minimal(ax)


def _plot_confidence(ax, name, risk, bins, theme):
    confidence = np.abs(risk - 0.5) * 2.0
    ax.hist(confidence, bins=bins, color=theme.ACCENT_PURPLE, alpha=0.85,
            edgecolor=theme.EDGE, linewidth=0.4)
    mean_conf = float(confidence.mean())
    med_conf = float(np.median(confidence))
    ax.axvline(mean_conf, color=theme.ACCENT_AMBER, ls='--', lw=1.2,
               label=f'mean = {mean_conf:.3f}')
    ax.axvline(med_conf,  color=theme.ACCENT_GREEN, ls=':', lw=1.2,
               label=f'median = {med_conf:.3f}')
    ax.set_xlabel('confidence')
    ax.set_ylabel('count')
    ax.legend(frameon=False, labelcolor=theme.TEXT)
    _minimal(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'],
                        default=['test'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=str, default='confidence_distribution.png')
    parser.add_argument('--out-dir', type=str, default='poster',
                        help='output subdir under outputs/ (default poster)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark',
                        help='color palette (default dark)')
    parser.add_argument('--threads', type=int, default=2,
                        help='torch CPU threads (default 2)')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    theme = light_theme if args.theme == 'light' else dark_theme

    import matplotlib.pyplot as plt
    theme.apply()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimalBitcoinGNN(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        hidden_dim=64,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, 'outputs', 'gnn_model.pt'),
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    temp_path = os.path.join(PROJECT_ROOT, 'outputs', 'temperature.pt')
    temperature = float(torch.load(temp_path, weights_only=True).get('temperature', 1.0))
    print(f"Using device={device}, T={temperature:.4f}, threads={args.threads}, splits={args.splits}")

    results = {}
    for split in args.splits:
        print(f"\nScoring {split} split...")
        risk, lbl = score_split(model, split, temperature, device, args.batch_size)
        print(f"  n={len(risk):,}")
        results[split] = (risk, lbl)

    bins = np.linspace(0, 1, 51)
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = args.out[:-4] if args.out.lower().endswith('.png') else args.out
    multi = len(args.splits) > 1

    def _out(split, suffix):
        stem = f'{base}_{split}' if multi else base
        return os.path.join(out_dir, f'{stem}_{suffix}.png')

    # Each panel is now its own standalone image (risk score + confidence).
    saved = []
    for split in args.splits:
        risk, lbl = results[split]
        name = split.capitalize()

        fig_r, ax_r = plt.subplots(figsize=(7.2, 4.8))
        _plot_risk(ax_r, name, risk, lbl, bins, theme)
        ax_r.set_title(f'GNN risk score — {name.lower()} (T = {temperature:.2f})',
                       color=theme.TEXT, fontsize=13)
        fig_r.tight_layout()
        p_r = _out(split, 'riskscore')
        fig_r.savefig(p_r, dpi=200, bbox_inches='tight')
        plt.close(fig_r)
        saved.append(p_r)

        fig_c, ax_c = plt.subplots(figsize=(7.2, 4.8))
        _plot_confidence(ax_c, name, risk, bins, theme)
        ax_c.set_title(f'GNN confidence — {name.lower()} (T = {temperature:.2f})',
                       color=theme.TEXT, fontsize=13)
        fig_c.tight_layout()
        p_c = _out(split, 'confidence')
        fig_c.savefig(p_c, dpi=200, bbox_inches='tight')
        plt.close(fig_c)
        saved.append(p_c)

    print()
    for p in saved:
        print(f"Saved {p}")

    summary = {'temperature': temperature, 'splits': args.splits}
    for split in args.splits:
        risk, lbl = results[split]
        summary[split] = _summary(risk, lbl)
    summary_path = os.path.join(out_dir, base + '.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
