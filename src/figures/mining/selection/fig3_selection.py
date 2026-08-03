#!/usr/bin/env python3
"""Fig 3: no answer-blind selector recovers the oracle-deployment gap.
(a) per-cell recovery for six deployable channels, all read at the same budget
    K=32, recovery = (selector - greedy)/(random-path oracle - greedy).
(b) the gap widens with sampling temperature: the ceiling rises, the vote does not.
Inputs are mined by mine_selection.py straight from the audit JSONs.
"""
import os, csv
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════ TUNABLE PARAMETERS ══════════════════════════════════════════
FIGSIZE     = (7.16, 2.72)
DPI         = 400
WIDTH_RATIO = [1.36, 1.0]
WSPACE      = 0.235
FS_TITLE    = 9.0
FS_LABEL    = 8.0
FS_TICK     = 7.4
FS_ANNOT    = 7.0
JITTER      = 0.135          # vertical scatter of per-cell dots
DOT_MS      = 3.4
MED_LEN     = 0.34           # half-length of the median tick

C_NEG   = "#B64342"          # a channel that loses ground
C_POS   = "#0F4D92"          # a channel that gains a little
C_MED   = "#1A1A1A"
C_FLAG  = "#C8860D"          # format-collapse cell, not a real recovery
C_GRID  = "#D8D8D8"
C_TEXT  = "#1A1A1A"
C_MUTE  = "#6E6E6E"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
rec  = list(csv.DictReader(open(os.path.join(HERE, "recovery.csv"))))
tmp  = list(csv.DictReader(open(os.path.join(HERE, "temperature.csv"))))

# Channels ordered as they appear in the paper: votes, then search, then verify.
ORDER = ["majority vote", "confidence", "MCTS vote", "hill climbing",
         "genetic search", "self-verification"]
SHORT = {"majority vote": "majority vote", "confidence": "confidence",
         "MCTS vote": "MCTS vote", "hill climbing": "hill climbing",
         "genetic search": "genetic search",
         "self-verification": "self-verification"}

by_ch = defaultdict(list)
for r in rec:
    by_ch[r["channel"]].append((float(r["recovery"]), r["flag"], r["cell"]))

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
gs  = fig.add_gridspec(1, 2, width_ratios=WIDTH_RATIO, wspace=WSPACE,
                       left=0.135, right=0.985, top=0.845, bottom=0.175)

# ══════════════ PANEL (a): per-cell recovery ════════════════════════════════
axa = fig.add_subplot(gs[0, 0])
axa.axvline(0, color=C_TEXT, lw=0.9, zorder=2)
rng = np.random.default_rng(0)

for i, ch in enumerate(ORDER):
    y0 = len(ORDER) - 1 - i
    vals = by_ch.get(ch, [])
    ok = [v for v, f, _ in vals if f == "ok"]
    axa.axhline(y0, color=C_GRID, lw=0.5, zorder=1)
    for v, flag, cell in vals:
        yy = y0 + rng.uniform(-JITTER, JITTER)
        if flag != "ok":
            axa.plot(v, yy, marker="D", ms=DOT_MS - 0.5, mfc="none",
                     mec=C_FLAG, mew=0.9, zorder=5)
        else:
            axa.plot(v, yy, marker="o", ms=DOT_MS, alpha=0.62, zorder=4,
                     mfc=C_POS if v > 0 else C_NEG, mec="none")
    if ok:
        med = float(np.median(ok))
        axa.plot([med, med], [y0 - MED_LEN, y0 + MED_LEN], color=C_MED,
                 lw=1.7, zorder=6, solid_capstyle="butt")
        axa.text(med, y0 + MED_LEN + 0.10, f"{med:+.1f}", fontsize=FS_ANNOT - 0.5,
                 ha="center", va="bottom", color=C_MED, zorder=7)

axa.set_yticks(range(len(ORDER)))
axa.set_yticklabels([SHORT[c] for c in reversed(ORDER)], fontsize=FS_TICK)
axa.set_xlim(-82, 90)
axa.set_ylim(-0.62, len(ORDER) - 0.30)
axa.set_xticks([-75, -50, -25, 0, 25, 50, 75])
axa.set_xlabel("recovery of the oracle-deployment gap (%)", fontsize=FS_LABEL,
               labelpad=1.5)
axa.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.5)
axa.set_title("(a)  every deployable channel is weak or negative",
              fontsize=FS_TITLE, loc="left", pad=4, color=C_TEXT)
axa.text(-80, -0.50, "one dot = one model$\\times$task cell, $K$=32; "
         "bar = median", fontsize=FS_ANNOT - 0.6, color=C_MUTE, ha="left")

# Name the only cells that look like a large recovery, and say why they are not.
axa.annotate("Qwen2.5-3B / ARC-ch:\ngreedy baseline broken\nby answer format",
             xy=(74.1, len(ORDER) - 1 - 0.30), xytext=(52, len(ORDER) - 1 - 1.30),
             fontsize=FS_ANNOT - 0.6, color=C_FLAG, ha="center", va="top",
             linespacing=1.25,
             arrowprops=dict(arrowstyle="-", color=C_FLAG, lw=0.55,
                             shrinkA=1, shrinkB=2))

# ══════════════ PANEL (b): the gap widens with temperature ══════════════════
axb = fig.add_subplot(gs[0, 1])
cells = defaultdict(list)
for r in tmp:
    cells[r["cell"]].append((float(r["temp"]), float(r["gap"])))

STYLE = {"3B/DART":       (C_NEG,  "o", "-",  2.0),
         "3B/DART-pool":  (C_NEG,  "s", "--", 1.7),
         "8B/HellaSwag":  (C_POS,  "^", "-",  1.7),
         "8B/ARC-ch":     (C_POS,  "v", "--", 1.5)}
ORDER_T = ["3B/DART-pool", "3B/DART", "8B/HellaSwag", "8B/ARC-ch"]

for cell in ORDER_T:
    pts = sorted(cells.get(cell, []))
    if not pts:
        continue
    col, mk, ls, lw = STYLE[cell]
    t = [p[0] for p in pts]; g = [p[1] for p in pts]
    axb.plot(t, g, color=col, marker=mk, ls=ls, lw=lw, ms=3.8, zorder=4)
    axb.text(t[-1] + 0.028, g[-1], cell, fontsize=FS_ANNOT - 0.4, va="center",
             ha="left", color=col)

axb.set_xlim(0.62, 1.62); axb.set_ylim(0, 52)
axb.set_xticks([0.7, 1.0, 1.3])
axb.set_yticks([0, 15, 30, 45])
axb.set_xlabel("sampling temperature", fontsize=FS_LABEL, labelpad=1.5)
axb.set_ylabel("selection gap (pp)", fontsize=FS_LABEL, labelpad=2.0)
axb.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.5)
axb.set_title("(b)  raising the ceiling does not fill the bag",
              fontsize=FS_TITLE, loc="left", pad=4, color=C_TEXT)
axb.text(0.655, 1.4, "gap = oracle@200 $-$ majority@200",
         fontsize=FS_ANNOT - 0.6, color=C_MUTE, ha="left")

OUT = HERE
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig3_selection.{ext}"), bbox_inches="tight",
                pad_inches=0.012)
print("wrote fig3_selection.pdf / .png")
