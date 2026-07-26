"""Light-mode color palette and matplotlib rcParams.

Mirrors the constant names exported by ``dark_theme`` so chart scripts can swap
themes by selecting the module (``theme = light_theme if ... else dark_theme``).
Colors are tuned for legibility on a white background (darker accents than the
dark theme, which assumes a near-black canvas).
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Backgrounds & neutrals
BG          = '#FFFFFF'
PANEL       = '#F4F6F8'
GRID        = '#D8DEE4'
TEXT        = '#1B2733'
TEXT_DIM    = '#5A6672'
EDGE        = '#C2CAD2'

# Class colors
BENIGN      = '#1976D2'   # blue
BENIGN_DARK = '#0D47A1'
CRIMINAL    = '#E64A19'   # warm orange-red
CRIMINAL_DARK = '#BF360C'

# Accents (for arrows, residuals, highlights)
ACCENT_PINK   = '#D81B60'
ACCENT_PURPLE = '#8E24AA'
ACCENT_AMBER  = '#F9A825'
ACCENT_GREEN  = '#2E7D32'
ACCENT_BLUE   = '#1565C0'

# Diverging / sequential colormap for heatmaps (white -> deep blue)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'light_blue_seq',
    ['#FFFFFF', '#E0F2FF', '#80D8FF', '#42A5F5', '#1E88E5', '#0D47A1'],
)


def apply():
    """Apply light theme rcParams globally."""
    mpl.rcParams.update({
        'figure.facecolor':     BG,
        'axes.facecolor':       BG,
        'savefig.facecolor':    BG,
        'savefig.edgecolor':    BG,
        'axes.edgecolor':       EDGE,
        'axes.labelcolor':      TEXT,
        'axes.titlecolor':      TEXT,
        'xtick.color':          TEXT_DIM,
        'ytick.color':          TEXT_DIM,
        'text.color':           TEXT,
        'grid.color':           GRID,
        'grid.alpha':           0.8,
        'axes.grid':            False,
        'legend.facecolor':     BG,
        'legend.edgecolor':     EDGE,
        'legend.labelcolor':    TEXT,
        'font.family':          'sans-serif',
        'font.sans-serif':      ['DejaVu Sans', 'Arial', 'Helvetica Neue'],
    })
