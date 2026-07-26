"""Generate dark-mode model architecture diagram for OptimalBitcoinGNN."""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_theme as dt

dt.apply()

# Section colors (dark-mode tuned)
INPUT  = '#1E2A3A'   # deep blue-gray
PREP   = '#3D2E12'   # dark amber for ghost-prep block
GAT    = '#1A3447'   # deep teal-blue
NORM   = '#1F3A2A'   # dark green
POOL   = '#3A1F45'   # dark purple
MLP    = '#3D2E10'   # dark orange-amber
TEMP   = '#3D2810'   # dark orange
BENIGN = dt.BENIGN_DARK
CRIM   = dt.CRIMINAL_DARK
RES1   = dt.ACCENT_PINK
RES2   = dt.ACCENT_PURPLE
DROP   = dt.CRIMINAL


def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False, edge=dt.EDGE, text_color=dt.TEXT):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=edge, linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            color=text_color)


def arrow(ax, x1, y1, x2, y2, color=dt.TEXT_DIM, lw=1.6, style='->'):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14, color=color, lw=lw,
        shrinkA=0, shrinkB=0,
    ))


def create_diagram():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(dt.BG)
    ax.set_facecolor(dt.BG)

    # ── Title ──
    ax.text(10, 10.5, 'OptimalBitcoinGNN', ha='center',
            fontsize=20, fontweight='bold', color=dt.ACCENT_BLUE)
    ax.text(10, 10.05,
            '3-layer GATv2 · ghost-node embeddings · DropEdge · hybrid readout · temperature scaling',
            ha='center', fontsize=11, color=dt.TEXT_DIM, style='italic')

    # ── Inputs ──
    draw_box(ax, 0.2, 7.3, 1.9, 1.0, 'Node features\n[N, 12]', INPUT, 9, True)
    draw_box(ax, 0.2, 5.8, 1.9, 1.0, 'Edge features\n[E, 3]',  INPUT, 9, True)
    draw_box(ax, 0.2, 4.3, 1.9, 1.0, 'edge_index\n[2, E]',     INPUT, 9, True)
    draw_box(ax, 0.2, 2.8, 1.9, 1.0, 'batch\n[N]',             INPUT, 9, True)

    # ── Ghost-node prep (NEW) ──
    draw_box(ax, 2.6, 6.8, 3.0, 2.0,
             'Ghost-node prep (NEW)\n'
             '• ghost_embedding (learnable, 12-d)\n'
             '• has_features flag (1-d)\n'
             '• node_type_emb {center, ghost} (8-d)\n'
             'Linear: 12 + 1 + 8 → 12',
             PREP, 8.2, True, edge=dt.ACCENT_AMBER)
    arrow(ax, 2.1, 7.8, 2.6, 7.8)

    # DropEdge (NEW)
    draw_box(ax, 2.6, 5.5, 3.0, 0.8,
             'DropEdge p=0.1 (train only, NEW)',
             DROP, 8.5, True, edge=dt.CRIMINAL_DARK)
    arrow(ax, 2.1, 6.3, 2.6, 5.9)
    arrow(ax, 2.1, 4.8, 2.6, 5.7)

    # ── GATv2 Layer 1 ──
    x1 = 6.0
    draw_box(ax, x1, 6.6, 2.4, 1.6,
             'GATv2Conv 1\n12 → 64, heads=4\nout: [N, 256]', GAT, 9.5, True)
    draw_box(ax, x1, 5.85, 2.4, 0.55, 'LayerNorm + ELU + Dropout', NORM, 8.2)
    arrow(ax, 5.6, 7.8, 6.0, 7.4)

    # ── GATv2 Layer 2 ──
    x2 = 9.1
    draw_box(ax, x2, 6.6, 2.4, 1.6,
             'GATv2Conv 2\n256 → 64, heads=4\nout: [N, 256]', GAT, 9.5, True)
    draw_box(ax, x2, 5.85, 2.4, 0.55, 'LayerNorm + ELU + Dropout', NORM, 8.2)
    arrow(ax, x1 + 2.4, 7.4, x2, 7.4)

    # Residual L1 -> L2
    ax.plot([x1 + 1.2, x1 + 1.2, x2 + 1.2, x2 + 1.2],
            [8.25, 8.95, 8.95, 8.25],
            color=RES1, lw=2.2, solid_capstyle='round')
    arrow(ax, x2 + 1.2, 8.30, x2 + 1.2, 8.20, color=RES1, lw=2.0)
    ax.text((x1 + x2) / 2 + 1.2, 9.15, 'residual (Linear 256→256)',
            ha='center', fontsize=8.5, color=RES1, fontweight='bold')

    # ── GATv2 Layer 3 ──
    x3 = 12.2
    draw_box(ax, x3, 6.6, 2.4, 1.6,
             'GATv2Conv 3\n256 → 32, heads=2\nout: [N, 64]', GAT, 9.5, True)
    draw_box(ax, x3, 5.85, 2.4, 0.55, 'LayerNorm + ELU + Dropout', NORM, 8.2)
    arrow(ax, x2 + 2.4, 7.4, x3, 7.4)

    # Initial-connection residual (NEW)
    ax.plot([3.6, 3.6, x3 + 1.2, x3 + 1.2],
            [6.7, 4.4, 4.4, 5.85],
            color=RES2, lw=2.2, solid_capstyle='round', linestyle='--')
    arrow(ax, x3 + 1.2, 5.95, x3 + 1.2, 5.85, color=RES2, lw=2.0)
    ax.text((3.6 + x3 + 1.2) / 2, 4.15,
            'initial → final residual (Linear 12→64) — NEW',
            ha='center', fontsize=8.8, color=RES2, fontweight='bold')

    # ── Hybrid readout (NEW) ──
    xR = 15.3
    draw_box(ax, xR, 8.0, 2.7, 0.8, 'center node h₀  [B, 64]',  POOL, 8.5, True)
    draw_box(ax, xR, 7.0, 2.7, 0.8, 'global mean pool  [B, 64]', POOL, 8.5, True)
    draw_box(ax, xR, 6.0, 2.7, 0.8, 'global max pool   [B, 64]', POOL, 8.5, True)
    arrow(ax, x3 + 2.4, 7.4, xR, 8.4)
    arrow(ax, x3 + 2.4, 7.4, xR, 7.4)
    arrow(ax, x3 + 2.4, 7.4, xR, 6.4)
    ax.text(xR + 1.35, 8.85, 'Hybrid readout (NEW)',
            ha='center', fontsize=9, color=dt.ACCENT_PURPLE, fontweight='bold')

    # Concat
    draw_box(ax, xR, 4.7, 2.7, 0.8, 'concat → [B, 192]', POOL, 9, True)
    arrow(ax, xR + 1.35, 6.0, xR + 1.35, 5.5)

    # Classifier head
    draw_box(ax, xR, 3.2, 2.7, 1.2,
             'MLP classifier\nLinear 192→64\nELU + Dropout(0.3)\nLinear 64→2',
             MLP, 8.8, True)
    arrow(ax, xR + 1.35, 4.7, xR + 1.35, 4.4)

    # Temperature scaling (NEW)
    draw_box(ax, xR, 1.95, 2.7, 0.85,
             'logits / T  (T=1.077)\ntemperature scaling (NEW)',
             TEMP, 8.5, True, edge=dt.ACCENT_AMBER)
    arrow(ax, xR + 1.35, 3.2, xR + 1.35, 2.8)

    # Output classes
    draw_box(ax, xR - 0.05, 0.55, 1.30, 0.95, 'Benign\nP(0)',   BENIGN, 9, True, text_color='white')
    draw_box(ax, xR + 1.45, 0.55, 1.30, 0.95, 'Criminal\nP(1)', CRIM,   9, True, text_color='white')
    arrow(ax, xR + 0.6, 1.95, xR + 0.6, 1.5)
    arrow(ax, xR + 2.1, 1.95, xR + 2.1, 1.5)

    # ── Legend ──
    lg_x, lg_y = 0.2, 1.5
    ax.text(lg_x, lg_y + 0.85, 'Legend', fontsize=10, fontweight='bold', color=dt.TEXT)
    items = [
        (PREP,   'Ghost-node prep (NEW)',           'box'),
        (DROP,   'DropEdge regularization (NEW)',   'box'),
        (RES1,   'Layer-1 → Layer-2 residual',      'line-solid'),
        (RES2,   'Initial → final residual (NEW)',  'line-dashed'),
        (POOL,   'Hybrid readout: center+mean+max', 'box'),
        (TEMP,   'Post-hoc temperature scaling',    'box'),
    ]
    for i, (c, label, kind) in enumerate(items):
        y = lg_y + 0.55 - i * 0.32
        if kind == 'line-solid':
            ax.plot([lg_x, lg_x + 0.35], [y, y], color=c, lw=2.5)
        elif kind == 'line-dashed':
            ax.plot([lg_x, lg_x + 0.35], [y, y], color=c, lw=2.5, linestyle='--')
        else:
            draw_box(ax, lg_x, y - 0.12, 0.35, 0.24, '', c, 1, edge=dt.EDGE)
        ax.text(lg_x + 0.5, y, label, fontsize=8.5, va='center', color=dt.TEXT)

    plt.tight_layout()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'outputs', 'model_architecture.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=dt.BG)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    create_diagram()
