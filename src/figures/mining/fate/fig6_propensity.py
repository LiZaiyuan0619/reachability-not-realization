#!/usr/bin/env python3
"""Fig 6: the gain lands where the base model was already close.
(a) among items the base could reach but not produce, the share that training
    turns into produced answers, binned by how often the base already hit them.
(b) the band of sampling temperatures that keeps the ceiling narrows after RLVR.
Panel (a) is mined by mine_fate.py; panel (b) is the published band table.
"""
import os, csv
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════ TUNABLE PARAMETERS ══════════════════════════════════════════
FIGSIZE  = (7.16, 2.30)
DPI      = 400
WRATIO   = [1.24, 1.0]
WSPACE   = 0.245
FS_TITLE = 8.8
FS_LABEL = 7.8
FS_TICK  = 7.2
FS_ANNOT = 6.9
MIN_N    = 5            # bins thinner than this are drawn hollow, never hidden

C_SR   = "#0F4D92"
C_DAPO = "#B64342"
C_OR1  = "#C8860D"
C_BASE = "#767676"
C_TEXT = "#1A1A1A"
C_MUTE = "#6E6E6E"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7.8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.75, "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
prop = list(csv.DictReader(open(os.path.join(HERE, "propensity.csv"))))
BINS = ["0-2%", "2-5%", "5-15%", "15-35%", ">35%"]
SERIES = [("SimpleRL", "dart", C_SR, "o", "-"),
          ("DAPO", "dart", C_DAPO, "s", "--"),
          ("OR1", "dart", C_OR1, "^", ":")]

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
gs  = fig.add_gridspec(1, 2, width_ratios=WRATIO, wspace=WSPACE,
                       left=0.070, right=0.988, top=0.845, bottom=0.235)

# ══════════════ (a) propensity by base hit rate ═════════════════════════════
axa = fig.add_subplot(gs[0, 0])
lut = {(r["pair"], r["task"], r["bucket"]): r for r in prop}
xs = np.arange(len(BINS))
for si, (pair, task, col, mk, ls) in enumerate(SERIES):
    ys, sizes, hollow = [], [], []
    for b in BINS:
        r = lut.get((pair, task, b))
        if r is None:
            ys.append(np.nan); sizes.append(0); hollow.append(True); continue
        ys.append(float(r["rate"]) * 100)
        sizes.append(int(r["total"]))
        hollow.append(int(r["total"]) < MIN_N)
    axa.plot(xs, ys, color=col, ls=ls, lw=1.5, zorder=3, label=pair)
    for xx, yy, nn, hol in zip(xs, ys, sizes, hollow):
        if np.isnan(yy):
            continue
        axa.plot(xx, yy, marker=mk, ms=4.2, zorder=5,
                 mfc="white" if hol else col, mec=col, mew=1.0)
        # Stagger the bin counts by series so they do not pile up where the
        # curves cross.
        axa.text(xx + (si - 1) * 0.115, yy + 4.6, f"{nn}",
                 fontsize=FS_ANNOT - 1.2, ha="center", va="bottom",
                 color=col, alpha=0.85, zorder=6)

axa.set_xticks(xs); axa.set_xticklabels(BINS, fontsize=FS_TICK)
axa.set_ylim(-6, 112); axa.set_yticks([0, 25, 50, 75, 100])
axa.set_xlabel("how often the base model already hit the item",
               fontsize=FS_LABEL, labelpad=1.5)
axa.set_ylabel("turned into a produced\nanswer (%)", fontsize=FS_LABEL,
               labelpad=1.5, linespacing=1.2)
axa.tick_params(labelsize=FS_TICK, length=2.2, pad=1.3)
axa.set_title("(a)  the gain lands where the base was already close",
              fontsize=FS_TITLE, loc="left", pad=4.0, color=C_TEXT)
axa.legend(fontsize=FS_ANNOT, loc="upper left", handlelength=1.5,
           labelspacing=0.26, borderpad=0.0, handletextpad=0.45)

# ══════════════ (b) the usable temperature band narrows ═════════════════════
# Source: appendix D.7.  A temperature is inside the band when the ceiling is
# within 5pp of its value at t=0.8.
axb = fig.add_subplot(gs[0, 1])
TEMPS = [0.8, 1.0, 1.3, 1.6]
BAND = [("Olmo base",     [0.740, 0.747, 0.713, 0.393], C_BASE, "-"),
        ("Olmo RL-zero",  [0.607, 0.580, 0.527, 0.393], C_DAPO, "--"),
        ("SimpleRL base", [0.940, 0.927, 0.933, 0.687], C_BASE, "-"),
        ("SimpleRL zero", [0.940, 0.927, 0.847, 0.660], C_SR,   "--")]

for name, ys, col, ls in BAND:
    v = [y * 100 for y in ys]
    thr = v[0] - 5.0
    inside = [y >= thr for y in v]
    axb.plot(TEMPS, v, color=col, ls=ls, lw=1.4, zorder=3,
             alpha=0.55 if col == C_BASE else 1.0)
    for t, y, ins in zip(TEMPS, v, inside):
        axb.plot(t, y, marker="o" if ins else "x", ms=4.4 if ins else 4.0,
                 mfc=col if ins else "none", mec=col, mew=1.1, zorder=5,
                 alpha=0.55 if col == C_BASE else 1.0)

axb.set_xlim(0.75, 1.66); axb.set_ylim(30, 108)
axb.legend([plt.Line2D([], [], color=c, ls=l, lw=1.4,
                       alpha=0.55 if c == C_BASE else 1.0)
            for _, _, c, l in BAND],
           [n for n, _, _, _ in BAND], fontsize=FS_ANNOT - 0.6,
           loc="lower left", ncol=2, handlelength=1.5, labelspacing=0.24,
           columnspacing=0.85, borderpad=0.0, handletextpad=0.42)
axb.set_xticks(TEMPS); axb.set_xticklabels(["0.8", "1.0", "1.3", "1.6"])
axb.set_yticks([40, 60, 80, 100])
axb.set_xlabel("sampling temperature", fontsize=FS_LABEL, labelpad=1.5)
axb.set_ylabel("reachable ceiling (%)", fontsize=FS_LABEL, labelpad=1.5)
axb.tick_params(labelsize=FS_TICK, length=2.2, pad=1.3)
axb.set_title("(b)  the usable band narrows", fontsize=FS_TITLE, loc="left",
              pad=4.0, color=C_TEXT)
axb.text(0.775, 106, "filled = inside the band", fontsize=FS_ANNOT - 0.8,
         color=C_MUTE, ha="left", va="top")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig6_propensity.{ext}"),
                bbox_inches="tight", pad_inches=0.012)
print("wrote fig6_propensity.pdf / .png")
