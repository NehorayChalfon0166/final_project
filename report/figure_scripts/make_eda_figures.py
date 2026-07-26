#!/usr/bin/env python3
"""Regenerate the two EDA figures for the project book, in the shared academic style.

Outputs vector PDFs to ``report/figures/``:
    fig_eda_composition.pdf   working-corpus composition (class + source, train vs test)
    fig_eda_degree.pdf        Elliptic++ transaction-graph in/out-degree distribution by class

Reads:
    src/features/output/train_dataset.csv, test_dataset.csv   (composition)
    data/elliptic/txs_edgelist.csv, txs_classes.csv           (degree distribution)

Run from anywhere (needs pandas, numpy, matplotlib):
    python report/figure_scripts/make_eda_figures.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import academic_style as A  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FEAT = os.path.join(PROJECT_ROOT, "src", "features", "output")
ELL = os.path.join(PROJECT_ROOT, "data", "elliptic")
FIG = os.path.join(PROJECT_ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
A.apply()


def fig_eda_composition():
    TR = pd.read_csv(os.path.join(FEAT, "train_dataset.csv"))
    TE = pd.read_csv(os.path.join(FEAT, "test_dataset.csv"))

    def cts(df):
        return (int((df.label == 0).sum()), int((df.label == 1).sum()),
                int((df.source == "realcats").sum()), int((df.source == "elliptic").sum()))
    trb, trc, trr, tre = cts(TR)
    teb, tec, ter, tee = cts(TE)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    splits = [f"Training\n({len(TR):,})", f"Test\n({len(TE):,})"]
    x = np.arange(2); w = 0.38

    for i, (ax, a, b, la, lb, ca, cb, title) in enumerate([
        (axes[0], [trb, teb], [trc, tec], "Benign", "Criminal", A.BENIGN, A.CRIMINAL, "Class composition"),
        (axes[1], [trr, ter], [tre, tee], "REAL-CATS", "Elliptic++", "#4f80ab", "#b3541e", "Source composition"),
    ]):
        ax.bar(x - w / 2, a, w, label=la, color=ca)
        ax.bar(x + w / 2, b, w, label=lb, color=cb)
        for xi, va, vb in zip(x, a, b):
            ax.text(xi - w / 2, va, f"{va:,}", ha="center", va="bottom", fontsize=6.5, color=A.SUBINK)
            ax.text(xi + w / 2, vb, f"{vb:,}", ha="center", va="bottom", fontsize=6.5, color=A.SUBINK)
        ax.set_title(title); ax.set_ylabel("Wallets")
        ax.set_xticks(x); ax.set_xticklabels(splits)
        ax.legend(loc="upper right"); ax.set_ylim(0, max(a) * 1.18)

    fig.tight_layout()
    A.save(fig, os.path.join(FIG, "fig_eda_composition.pdf"))


def fig_eda_degree():
    edges = pd.read_csv(os.path.join(ELL, "txs_edgelist.csv"))
    cls = pd.read_csv(os.path.join(ELL, "txs_classes.csv"))  # class: 1 illicit, 2 licit, 3 unknown
    nodes = pd.Index(np.union1d(edges.txId1.unique(), edges.txId2.unique()))
    indeg = edges.txId2.value_counts().reindex(nodes, fill_value=0)
    outdeg = edges.txId1.value_counts().reindex(nodes, fill_value=0)
    cmap = cls.set_index("txId")["class"].reindex(nodes)

    be = [-.5, .5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 128.5, np.inf]
    labels = ["0", "1", "2", "3–4", "5–8", "9–16", "17–32", "33–64", "65–128", "129+"]

    def frac(a):
        h, _ = np.histogram(a, bins=be)
        return h / max(h.sum(), 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    x = np.arange(len(labels)); w = 0.4
    for ax, s, t in [(axes[0], indeg, "In-degree"), (axes[1], outdeg, "Out-degree")]:
        ill = s[cmap.values == 1].values
        lic = s[cmap.values == 2].values
        ax.bar(x - w / 2, frac(ill), w, color=A.CRIMINAL, label="Illicit")
        ax.bar(x + w / 2, frac(lic), w, color=A.BENIGN, label="Licit")
        ax.set_yscale("log"); ax.set_ylim(0.02, 100)
        ax.set_title(t, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.8)
        ax.set_xlabel("Degree")
    axes[0].set_ylabel("% of class's nodes (log)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Elliptic++ transaction-graph degree distribution by class", fontsize=9.5, y=1.03)
    fig.tight_layout()
    A.save(fig, os.path.join(FIG, "fig_eda_degree.pdf"))


if __name__ == "__main__":
    fig_eda_composition()
    fig_eda_degree()
    print("wrote fig_eda_composition.pdf and fig_eda_degree.pdf to", FIG)
