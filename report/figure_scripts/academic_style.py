"""Shared "academic paper" matplotlib style for the project-book figures.

A restrained, publication-oriented look: serif type that matches the LaTeX body,
a muted monochrome-blue palette (our model is the darkest so the eye lands on it),
thin spines, hairline grids, and no chart-junk. Every figure is written as a
vector PDF so it stays crisp at any size in the report.

Import and call ``apply()`` once at the top of a figure script, then use the
exported colours / helpers.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette ----
# Neutrals
INK      = "#1a1a1a"   # primary text / axes
SUBINK   = "#5f6b7a"   # secondary text / ticks
GRID     = "#dfe3e8"   # hairline grid
PANEL    = "#f4f6f8"

# Model palette — monochrome blue ramp, "ours" is darkest for emphasis.
OPTIMAL  = "#16324f"   # OptimalBitcoinGNN (ours)
XGB      = "#4f80ab"   # XGBoost (tabular baseline)
GCN      = "#9fb0c3"   # Basic GCN (graph baseline)

# Sparingly used accents
ACCENT   = "#b3541e"   # terracotta — highlight a single element
BENIGN   = "#4f80ab"
CRIMINAL = "#16324f"

MODEL_COLORS = {
    "OptimalGNN": OPTIMAL,
    "Optimal GNN": OPTIMAL,
    "XGBoost": XGB,
    "BasicGCN": GCN,
    "Basic GCN": GCN,
}

# Sequential colormap for confusion-matrix heatmaps (white -> deep navy).
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "acad_blue", ["#ffffff", "#dbe5ef", "#9fb9d4", "#4f80ab", "#16324f"]
)


def apply() -> None:
    """Apply the academic rcParams globally."""
    mpl.rcParams.update({
        # type
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # colours
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": SUBINK,
        "ytick.color": SUBINK,
        # spines / ticks
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # grid
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        # figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # embed TrueType so text stays selectable
        "ps.fonttype": 42,
    })


def save(fig, path: str) -> None:
    """Save a figure as a vector PDF (and nothing else)."""
    fig.savefig(path)
    plt.close(fig)
