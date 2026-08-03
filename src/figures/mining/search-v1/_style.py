"""Shared style + knobs for Fig 2 (Search claim) v1 broad-net.

All human-tunable parameters live HERE and at the top of each plot script.
Edit these to globally adjust color / size / fonts across every variant.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ======================= GLOBAL TUNABLE KNOBS =======================
# --- canvas / dpi ---
DPI_PREVIEW = 150
DPI_EXPORT  = 300

# --- fonts (pt) ---
FS_TITLE   = 11
FS_LABEL   = 9
FS_TICK    = 8
FS_ANNOT   = 7
FS_PANEL   = 12   # (a)(b)(c) panel letters

# --- color families (colorblind-safe) ---
# diverging map for heatmap: white(0) -> warm(positive gap). We clip at 0 (all data >=0).
CMAP_GAP   = "YlOrRd"          # try also: "Oranges", "magma_r", "cividis"
C_RANDOM   = "#D1495B"         # signal red  — random path (the hero method)
C_MCTS     = "#30638E"         # neutral blue — structured search / MCTS
C_SAMPLING = "#8AA29E"         # muted green-grey — sampling oracle
C_HC       = "#9BB1C9"         # light blue — hillclimb
C_GA       = "#5E86AD"         # mid blue — genetic
C_BASE     = "#B0B0B0"         # grey — deploy baseline
C_FLAG     = "#000000"         # marker for format-collapse cells

# --- element sizes ---
LW_CURVE   = 2.2
MS_MARKER  = 5
BAR_W      = 0.26
GRID_A     = 0.25             # grid alpha

# --- heatmap ---
HEAT_VMIN  = 0
HEAT_VMAX  = 50              # cap gap color scale (max real gap ~50)
HEAT_TEXT  = True            # annotate each cell with the gap value

# ===================================================================


def apply_rc():
    mpl.rcParams.update({
        "figure.dpi": DPI_PREVIEW,
        "savefig.dpi": DPI_EXPORT,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
        "legend.frameon": False,
        "axes.grid": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.08, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="right")
