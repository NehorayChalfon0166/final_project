"""Dark-mode color palette and matplotlib rcParams used across all chart scripts."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Backgrounds & neutrals
BG          = '#0F1419'
PANEL       = '#161B22'
GRID        = '#2D333B'
TEXT        = '#E6EDF3'
TEXT_DIM    = '#8B949E'
EDGE        = '#30363D'

# Class colors
BENIGN      = '#4FC3F7'   # cyan-blue
BENIGN_DARK = '#0288D1'
CRIMINAL    = '#FF7043'   # warm orange
CRIMINAL_DARK = '#D84315'

# Accents (for arrows, residuals, highlights)
ACCENT_PINK   = '#F06292'
ACCENT_PURPLE = '#BA68C8'
ACCENT_AMBER  = '#FFCA28'
ACCENT_GREEN  = '#66BB6A'
ACCENT_BLUE   = '#42A5F5'

# Diverging / sequential colormaps for heatmaps
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'dark_blue_seq',
    ['#0F1419', '#1A237E', '#3949AB', '#42A5F5', '#80D8FF', '#E0F7FA'],
)


def apply():
    """Apply dark theme rcParams globally."""
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
        'grid.alpha':           0.6,
        'axes.grid':            False,
        'legend.facecolor':     PANEL,
        'legend.edgecolor':     EDGE,
        'legend.labelcolor':    TEXT,
        'font.family':          'sans-serif',
        'font.sans-serif':      ['DejaVu Sans', 'Arial', 'Helvetica Neue'],
    })
