#!/usr/bin/env python3
"""Regenerate every figure in the project book in a consistent academic style.

Outputs vector PDFs to ``report/figures/``:

    Always (only need saved JSON / history, no ML deps):
        fig_method_overview.pdf     system / method schematic
        fig_architecture.pdf        OptimalBitcoinGNN architecture schematic
        fig_training_curves.pdf     train loss + val F1 vs epoch
        fig_confusion_panel.pdf     3-model confusion matrices
        fig_metric_comparison.pdf   grouped bars over 5 metrics x 3 models
        fig_feature_importance.pdf  GNN gradient-saliency importances

    Only when torch + torch_geometric + xgboost are importable (i.e. run inside
    the project venv), because they re-score the cached test ego-graphs:
        fig_roc.pdf                 3-model ROC overlay
        fig_reliability.pdf         GNN reliability diagram (after temperature)

Run from anywhere:
    python report/figure_scripts/make_figures.py
Optional:
    python report/figure_scripts/make_figures.py --no-score   # skip ML figures
    python report/figure_scripts/make_figures.py --only roc reliability
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)

import academic_style as A  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "outputs")
EVAL = os.path.join(OUT, "evaluation")
FIG = os.path.join(PROJECT_ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


# =====================================================================
#  Data-driven figures (no ML dependencies)
# =====================================================================
def fig_training_curves():
    h = _load(os.path.join(OUT, "gnn_training_history.json"))
    hist = h["history"]
    loss = np.asarray(hist["train_loss"], float)
    valf1 = np.asarray(hist["val_f1"], float)
    epochs = np.arange(1, len(loss) + 1)
    best = int(h.get("best_epoch", int(np.argmax(valf1)) + 1))

    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))
    l1, = ax1.plot(epochs, loss, color=A.INK, lw=1.6, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_xlim(1, len(loss))

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    l2, = ax2.plot(epochs, valf1, color=A.XGB, lw=1.6, ls="--",
                   label="Validation F$_1$")
    ax2.set_ylabel("Validation F$_1$")
    ax2.set_ylim(min(valf1) - 0.03, 1.0)

    ax1.axvline(best, color=A.ACCENT, lw=1.0, ls=":")
    ax1.annotate(f"best epoch {best}\n(early stopping)",
                 xy=(best, ax1.get_ylim()[1]),
                 xytext=(best + 3, ax1.get_ylim()[1] * 0.92),
                 fontsize=7.5, color=A.ACCENT, va="top")
    ax1.legend(handles=[l1, l2], loc="upper center", ncol=2)
    A.save(fig, os.path.join(FIG, "fig_training_curves.pdf"))
    print("  wrote fig_training_curves.pdf")


def _confusion(ax, cm, title):
    cm = np.asarray(cm, float)               # rows = true, cols = pred
    row = cm / cm.sum(axis=1, keepdims=True)  # row-normalized for colour
    im = ax.imshow(row, cmap=A.SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)                            # no grid lines across the cells
    labels = ["Benign", "Criminal"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    # thin white separators between cells (drawn only on the boundaries)
    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            ax.text(j, i, f"{val:,}\n{row[i, j]*100:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if row[i, j] > 0.55 else A.INK)
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, which="both")
    return im


def fig_confusion_panel():
    c = _load(os.path.join(EVAL, "three_model_comparison.json"))["models"]
    order = [("BasicGCN", "Basic GCN"), ("XGBoost", "XGBoost"),
             ("OptimalGNN", "Optimal GNN (ours)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    im = None
    for ax, (key, name) in zip(axes, order):
        m = c[key]
        im = _confusion(ax, m["confusion_matrix"], f"{name}\nF$_1$ = {m['f1']:.3f}")
    axes[0].set_ylabel("True")
    fig.subplots_adjust(left=0.08, right=0.9, wspace=0.35, bottom=0.16, top=0.82)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Row-normalized rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    A.save(fig, os.path.join(FIG, "fig_confusion_panel.pdf"))
    print("  wrote fig_confusion_panel.pdf")


def fig_metric_comparison():
    c = _load(os.path.join(EVAL, "three_model_comparison.json"))["models"]
    metrics = [("f1", "F$_1$"), ("accuracy", "Accuracy"),
               ("precision", "Precision"), ("recall", "Recall"),
               ("roc_auc", "ROC-AUC")]
    models = [("OptimalGNN", "Optimal GNN (ours)", A.OPTIMAL),
              ("XGBoost", "XGBoost", A.XGB),
              ("BasicGCN", "Basic GCN", A.GCN)]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (key, name, col) in enumerate(models):
        vals = [c[key][mk] for mk, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=col,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, color=A.SUBINK)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Score (held-out test set)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              columnspacing=1.4, handlelength=1.3)
    ax.grid(axis="x", visible=False)
    A.save(fig, os.path.join(FIG, "fig_metric_comparison.pdf"))
    print("  wrote fig_metric_comparison.pdf")


def fig_feature_importance():
    fi = _load(os.path.join(EVAL, "feature_importance.json"))["feature_importance"]
    items = sorted(fi.items(), key=lambda kv: kv[1])
    names = [k.replace("_log", "").replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    colors = [A.OPTIMAL] * len(vals)
    colors[-1] = A.ACCENT
    colors[-2] = A.ACCENT
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7, color=A.SUBINK)
    ax.set_xlabel("Gradient-saliency importance (normalized)")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(axis="y", visible=False)
    A.save(fig, os.path.join(FIG, "fig_feature_importance.pdf"))
    print("  wrote fig_feature_importance.pdf")


# =====================================================================
#  Schematic figures (pure matplotlib, vector)
# =====================================================================
def _box(ax, cx, cy, w, h, text, fc="white", ec=A.INK, fs=8.5, lw=0.9, tc=A.INK):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, color=A.INK, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, shrinkA=1, shrinkB=1))


def _stage(ax, cx, y, bw, bh, title, sub, fc, ec, tc, sc, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - bw / 2, y - bh / 2), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=1.2",
        linewidth=1.0, edgecolor=ec, facecolor=fc))
    fs_t = 8.2 if len(title) <= 15 else 7.0   # shrink long titles so they fit
    ax.text(cx, y + 1.9, title, ha="center", va="center", fontsize=fs_t,
            color=tc, fontweight="bold" if bold else "normal")
    ax.text(cx, y - 2.5, sub, ha="center", va="center", fontsize=6.6,
            color=sc, style="italic")


def fig_method_overview():
    fig, ax = plt.subplots(figsize=(9.0, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    # (title, subtitle, facecolor, emphasized?)
    stages = [
        ("Bitcoin address", "input", A.PANEL, False),
        ("Fetch history", "mempool.space", "white", False),
        ("Build ego-graph", "1-hop neighbours", "white", False),
        ("OptimalBitcoinGNN", "3 $\\times$ GATv2", "#dce6f2", True),
        ("Temperature", "calibration", "white", False),
        ("Risk verdict", "P[criminal]", A.OPTIMAL, True),
    ]
    n = len(stages)
    bw, bh = 14.6, 14
    gap = (100 - n * bw) / (n + 1)
    xs = [gap + i * (bw + gap) + bw / 2 for i in range(n)]
    y = 26
    for (title, sub, fc, emph), cx in zip(stages, xs):
        dark = fc == A.OPTIMAL
        _stage(ax, cx, y, bw, bh, title, sub, fc,
               ec=A.OPTIMAL if (dark or emph) else A.INK,
               tc="white" if dark else A.INK,
               sc="#d6dfea" if dark else A.SUBINK, bold=emph)
    for i in range(n - 1):
        _arrow(ax, xs[i] + bw / 2, y, xs[i + 1] - bw / 2, y, lw=1.2)
    # training-data source feeding the model from below (dashed, centred on GNN)
    gx = xs[3]
    tw, tyc, tyh = 54, 8, 7
    ax.add_patch(FancyBboxPatch(
        (gx - tw / 2, tyc - tyh / 2), tw, tyh,
        boxstyle="round,pad=0.02,rounding_size=1.0",
        linewidth=1.0, edgecolor=A.ACCENT, facecolor="#f7f2ec",
        linestyle=(0, (4, 2))))
    ax.text(gx, tyc, "Trained offline on REAL-CATS + Elliptic++  (balanced 80/20 split)",
            ha="center", va="center", fontsize=7.0, color=A.INK)
    ax.add_patch(FancyArrowPatch(
        (gx, tyc + tyh / 2), (gx, y - bh / 2), arrowstyle="-|>",
        mutation_scale=11, linewidth=1.1, color=A.ACCENT, linestyle=(0, (4, 2))))
    A.save(fig, os.path.join(FIG, "fig_method_overview.pdf"))
    print("  wrote fig_method_overview.pdf")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.0, 7.4))
    ax.set_xlim(0, 104); ax.set_ylim(0, 100); ax.axis("off")
    cx, bw = 40, 56
    # (y, text, facecolor, fontsize)
    rows = [
        (94, "Ego-graph input\nnode features (12)  +  edge features (3)", A.PANEL, 8.4),
        (83, "Input prep:  ghost embedding + node-type (8)\nconcat (21) $\\rightarrow$ project to 12", "white", 7.6),
        (70, "GATv2 layer 1 — 4 heads $\\rightarrow$ 256\nLayerNorm · ELU · dropout 0.2", "#e9eef4", 7.6),
        (57, "GATv2 layer 2 — 4 heads $\\rightarrow$ 256\nLayerNorm · ELU · dropout 0.2", "#e9eef4", 7.6),
        (44, "GATv2 layer 3 — 2 heads $\\rightarrow$ 64\nLayerNorm · ELU · dropout 0.1", "#e9eef4", 7.6),
        (32, "Hybrid readout\ncenter $\\oplus$ mean-pool $\\oplus$ max-pool  (192)", "white", 8.0),
        (20, "MLP classifier  192 $\\rightarrow$ 64 $\\rightarrow$ 2", "white", 8.2),
        (9,  "Softmax $\\rightarrow$ P[criminal]", A.OPTIMAL, 8.2),
    ]
    bh = 8.6
    for y, txt, fc, fs in rows:
        tc = "white" if fc == A.OPTIMAL else A.INK
        _box(ax, cx, y, bw, bh, txt, fc=fc, fs=fs, tc=tc)
    for i in range(len(rows) - 1):
        _arrow(ax, cx, rows[i][0] - bh / 2, cx, rows[i + 1][0] + bh / 2)

    right = cx + bw / 2  # right edge of the stack (= 68)
    # (a) short residual: layer 1 -> layer 2
    xr1 = right + 6
    for seg in [((right, 70), (xr1, 70)), ((xr1, 70), (xr1, 57))]:
        ax.add_patch(FancyArrowPatch(*seg, arrowstyle="-", color=A.ACCENT, lw=1.0))
    _arrow(ax, xr1, 57, right, 57, color=A.ACCENT, lw=1.0)
    ax.text(xr1 + 2.4, 63.5, "residual", rotation=90, ha="center", va="center",
            fontsize=6.2, color=A.ACCENT)
    # (b) initial-connection residual: input-prep (x_init) -> after layer 3
    xr2 = right + 18
    for seg in [((right, 83), (xr2, 83)), ((xr2, 83), (xr2, 44))]:
        ax.add_patch(FancyArrowPatch(*seg, arrowstyle="-", color=A.ACCENT, lw=1.1))
    _arrow(ax, xr2, 44, right, 44, color=A.ACCENT, lw=1.1)
    ax.text(xr2 + 2.6, 63.5, "initial-connection residual", rotation=90,
            ha="center", va="center", fontsize=6.2, color=A.ACCENT)
    # left-side note: DropEdge feeds layers 1-2
    ax.text(cx - bw / 2 - 4, 63.5, "DropEdge 0.1", rotation=90, ha="center",
            va="center", fontsize=6.6, color=A.SUBINK)
    A.save(fig, os.path.join(FIG, "fig_architecture.pdf"))
    print("  wrote fig_architecture.pdf")


# =====================================================================
#  Score-dependent figures (need the project venv: torch, PyG, xgboost)
# =====================================================================
def _roc_np(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(np.float64)
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0.0], np.cumsum(y) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - y) / N])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def _compute_scores():
    """Re-score the shared test set for all three models. Cached to npz."""
    cache = os.path.join(EVAL, "test_scores.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: (d[f"y_{k}"], d[f"p_{k}"]) for k in ("opt", "xgb", "gcn")}, \
            float(d["temperature"])

    import torch
    from torch_geometric.loader import DataLoader
    from src.graph.config import (FEATURE_COLUMNS, NUM_EDGE_FEATURES,
                                  NUM_NODE_FEATURES, TEST_DATASET_PATH,
                                  TRAIN_DATASET_PATH)
    from src.graph.dataloader import EgoGraphDataset
    from src.models.optimal_gnn import OptimalBitcoinGNN
    from src.models.utils import get_center_labels
    from src.baselines.gcn_baseline import BasicGCN

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    test_ds = EgoGraphDataset(split="test")
    train_ds = EgoGraphDataset(split="train")
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    temp = 1.0
    tpath = os.path.join(OUT, "temperature.pt")
    if os.path.exists(tpath):
        t = torch.load(tpath, weights_only=True)
        temp = float(t.get("temperature", 1.0)) if isinstance(t, dict) else float(t)

    def score_graph(model, is_opt):
        model.eval(); probs, labels = [], []
        with torch.no_grad():
            for b in loader:
                out = model(b.x, b.edge_index, b.edge_attr, b.batch) if is_opt \
                    else model(b.x, b.edge_index, b.batch)
                probs.append(torch.softmax(out, 1)[:, 1].numpy())
                labels.append(get_center_labels(b).numpy())
        return np.concatenate(labels).astype(int), np.concatenate(probs)

    opt = OptimalBitcoinGNN(num_node_features=NUM_NODE_FEATURES,
                            num_edge_features=NUM_EDGE_FEATURES, hidden_dim=64)
    opt.load_state_dict(torch.load(os.path.join(OUT, "gnn_model.pt"),
                                   map_location=device, weights_only=True))
    y_opt, p_opt = score_graph(opt, True)

    gcn = BasicGCN(num_node_features=NUM_NODE_FEATURES, hidden_dim=64, dropout=0.3)
    gcn.load_state_dict(torch.load(os.path.join(OUT, "baseline", "gcn_model.pt"),
                                   map_location=device, weights_only=True))
    y_gcn, p_gcn = score_graph(gcn, False)

    # XGBoost re-fit on the same train intersection (matches the eval protocol).
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    tr = pd.read_csv(TRAIN_DATASET_PATH)
    te = pd.read_csv(TEST_DATASET_PATH)
    tr = tr[tr["address"].isin(set(train_ds.addresses))].reset_index(drop=True)
    te = te[te["address"].isin(set(test_ds.addresses))].reset_index(drop=True)
    sc = StandardScaler()
    xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        eval_metric="logloss", verbosity=0, random_state=42)
    xgb.fit(sc.fit_transform(tr[FEATURE_COLUMNS].values), tr["label"].values)
    p_xgb = xgb.predict_proba(sc.transform(te[FEATURE_COLUMNS].values))[:, 1]
    y_xgb = te["label"].values.astype(int)

    np.savez(cache, y_opt=y_opt, p_opt=p_opt, y_gcn=y_gcn, p_gcn=p_gcn,
             y_xgb=y_xgb, p_xgb=p_xgb, temperature=temp)
    print(f"  cached per-sample scores -> {cache}")
    return {"opt": (y_opt, p_opt), "xgb": (y_xgb, p_xgb), "gcn": (y_gcn, p_gcn)}, temp


def fig_roc():
    scores, _ = _compute_scores()
    series = [("Optimal GNN (ours)", A.OPTIMAL, 2.0, "opt"),
              ("XGBoost", A.XGB, 1.4, "xgb"),
              ("Basic GCN", A.GCN, 1.4, "gcn")]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    rows = []
    for name, col, lw, key in series:
        yt, pr = scores[key]
        fpr, tpr, auc = _roc_np(yt, pr)
        rows.append((auc, name, col, lw, fpr, tpr))
    for auc, name, col, lw, fpr, tpr in sorted(rows, reverse=True):
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"{name}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=A.SUBINK, ls="--", lw=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.003)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    A.save(fig, os.path.join(FIG, "fig_roc.pdf"))
    print("  wrote fig_roc.pdf")


def fig_reliability():
    scores, temp = _compute_scores()
    # Apply temperature to the GNN's P[criminal] via the logit.
    y, p = scores["opt"]
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    p_cal = 1.0 / (1.0 + np.exp(-logit / temp))

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p_cal, bins) - 1
    idx = np.clip(idx, 0, 9)
    conf, acc, weight = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        conf.append(p_cal[m].mean()); acc.append(y[m].mean()); weight.append(m.mean())
    conf, acc, weight = map(np.asarray, (conf, acc, weight))
    ece = float(np.sum(weight * np.abs(acc - conf)))

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], color=A.SUBINK, ls="--", lw=0.9, label="Perfect calibration")
    ax.plot(conf, acc, "o-", color=A.OPTIMAL, lw=1.6, ms=4,
            label=f"GNN (T = {temp:.2f})")
    ax.bar(bins[:-1] + 0.05, weight, width=0.09, color=A.GCN, alpha=0.35,
           label="Fraction of samples")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P[criminal]")
    ax.set_ylabel("Empirical fraction criminal")
    ax.text(0.04, 0.92, f"ECE = {ece:.3f}", fontsize=8.5, color=A.INK)
    ax.legend(loc="lower right")
    A.save(fig, os.path.join(FIG, "fig_reliability.pdf"))
    print("  wrote fig_reliability.pdf")


# =====================================================================
DATA_FIGS = {
    "method": fig_method_overview,
    "architecture": fig_architecture,
    "training": fig_training_curves,
    "confusion": fig_confusion_panel,
    "metrics": fig_metric_comparison,
    "importance": fig_feature_importance,
}
SCORE_FIGS = {"roc": fig_roc, "reliability": fig_reliability}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-score", action="store_true",
                    help="skip figures that require re-scoring the test set")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: " + ", ".join(list(DATA_FIGS) + list(SCORE_FIGS)))
    args = ap.parse_args()

    A.apply()
    wanted = args.only
    print("Academic figures ->", FIG)

    for name, fn in DATA_FIGS.items():
        if wanted and name not in wanted:
            continue
        fn()

    if args.no_score:
        print("Skipping score-dependent figures (--no-score).")
        return
    for name, fn in SCORE_FIGS.items():
        if wanted and name not in wanted:
            continue
        try:
            fn()
        except ImportError as e:
            print(f"  [skip] {name}: needs the project venv (torch/PyG/xgboost). {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
