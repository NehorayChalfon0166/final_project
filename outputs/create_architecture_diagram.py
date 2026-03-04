import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_model_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    input_color = '#E3F2FD'
    gat_color = '#BBDEFB'
    norm_color = '#C8E6C9'
    mlp_color = '#FFE0B2'
    output_color = '#F8BBD9'
    arrow_color = '#455A64'

    def draw_box(x, y, width, height, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor=color, edgecolor='#37474F', linewidth=2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # Title
    ax.text(9, 7.5, 'OptimalBitcoinGNN', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1A237E')

    # Input Section (left side)
    x_input = 0.3
    draw_box(x_input, 4.8, 1.6, 1.2, 'Node\nFeatures', input_color, 9, True)
    draw_box(x_input, 2.5, 1.6, 1.2, 'Edge\nFeatures', input_color, 9, True)

    # Arrows from inputs converging
    draw_arrow(1.9, 5.4, 2.5, 4.2)
    draw_arrow(1.9, 3.1, 2.5, 4.2)

    # GAT Layer 1
    x_gat1 = 2.5
    draw_box(x_gat1, 3.2, 2.0, 2.0, 'GATv2Conv\nLayer 1', gat_color, 10, True)
    draw_box(x_gat1, 2.5, 2.0, 0.6, 'LN+ELU+DO', norm_color, 8)

    draw_arrow(4.5, 4.2, 5.1, 4.2)

    # GAT Layer 2
    x_gat2 = 5.1
    draw_box(x_gat2, 3.2, 2.0, 2.0, 'GATv2Conv\nLayer 2', gat_color, 10, True)
    draw_box(x_gat2, 2.5, 2.0, 0.6, 'LN+ELU+DO', norm_color, 8)

    # Residual connection - curved over the top
    ax.plot([3.5, 3.5, 6.1, 6.1], [5.2, 5.8, 5.8, 5.2],
            color='#E91E63', lw=2, solid_capstyle='round')
    ax.annotate('', xy=(6.1, 5.2), xytext=(6.1, 5.5),
               arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2))
    ax.text(4.8, 6.1, '+', fontsize=14, ha='center', va='center',
            color='#E91E63', fontweight='bold')

    draw_arrow(7.1, 4.2, 7.7, 4.2)

    # GAT Layer 3
    x_gat3 = 7.7
    draw_box(x_gat3, 3.2, 2.0, 2.0, 'GATv2Conv\nLayer 3', gat_color, 10, True)
    draw_box(x_gat3, 2.5, 2.0, 0.6, 'LN+ELU+DO', norm_color, 8)

    draw_arrow(9.7, 4.2, 10.3, 4.2)

    # Center node extraction
    x_extract = 10.3
    draw_box(x_extract, 3.4, 1.8, 1.6, 'Center\nNode', '#E1BEE7', 10, True)

    draw_arrow(12.1, 4.2, 12.7, 4.2)

    # Classification Head
    x_mlp = 12.7
    draw_box(x_mlp, 3.4, 1.8, 1.6, 'MLP\nHead', mlp_color, 10, True)

    draw_arrow(14.5, 4.2, 15.1, 4.2)

    # Output
    x_out = 15.1
    draw_box(x_out, 4.6, 1.8, 1.0, 'Benign', '#C8E6C9', 10, True)
    draw_box(x_out, 2.9, 1.8, 1.0, 'Criminal', '#FFCDD2', 10, True)

    # Arrows to outputs
    ax.plot([15.1, 15.1], [4.2, 5.1], color=arrow_color, lw=2)
    ax.plot([15.1, 15.1], [4.2, 3.4], color=arrow_color, lw=2)
    ax.annotate('', xy=(15.1, 5.1), xytext=(15.1, 4.8),
               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    ax.annotate('', xy=(15.1, 3.4), xytext=(15.1, 3.7),
               arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    plt.tight_layout()
    plt.savefig('outputs/model_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Diagram saved to outputs/model_architecture.png")

if __name__ == '__main__':
    create_model_architecture_diagram()
