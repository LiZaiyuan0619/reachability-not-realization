#!/usr/bin/env python3
"""Fig 2a (HERO): budget small-multiples. Real per-K values, zero interpolation.
Each panel = one model/task cell. Three lines: random oracle (climbs),
MCTS oracle (saturates early), deployable majority (pinned near base)."""
import os, csv
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── MANDATORY editable-text rcParams ────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 10, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 1.0, "legend.frameon": False,
})

C_RAND = "#0F4D92"   # random oracle — hero (deep blue)
C_MCTS = "#B64342"   # MCTS oracle — comparison (red)
C_MAJ  = "#767676"   # deployable majority — neutral

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "budget_real.csv"))))
cells, order = defaultdict(list), []
for r in rows:
    if r["cell"] not in cells: order.append(r["cell"])
    cells[r["cell"]].append(r)

fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
axes = axes.ravel()

for idx, label in enumerate(order):
    ax = axes[idx]
    d = sorted(cells[label], key=lambda r: int(r["K"]))
    K = np.array([int(r["K"]) for r in d])
    rand = np.array([float(r["rand_oracle"])   for r in d])
    mcts = np.array([float(r["mcts_oracle"])   for r in d])
    maj  = np.array([float(r["rand_majority"]) for r in d])
    orig = float(d[0]["orig"])
    # shaded gap between random-oracle and MCTS-oracle = "search has no edge"
    ax.fill_between(K, mcts, rand, color=C_RAND, alpha=0.08, zorder=1)
    ax.plot(K, rand, color=C_RAND, lw=2.4, marker="o", ms=4.5, zorder=4, label="Random-path oracle")
    ax.plot(K, mcts, color=C_MCTS, lw=2.2, ls="--", marker="s", ms=4, zorder=3, label="MCTS oracle")
    ax.plot(K, maj,  color=C_MAJ,  lw=1.8, ls=":", marker="^", ms=3.5, zorder=2, label="Deployable vote")
    ax.axhline(orig, color=C_MAJ, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 64, 200]); ax.set_xticklabels(["1", "8", "64", "200"])
    ax.set_ylim(-4, 104); ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=8.5)
    ax.set_title(label, fontsize=9.5, pad=3)
    ax.margins(x=0.02)

# ── 8th cell = legend + reading guide ───────────────────────────────────────
axL = axes[7]; axL.set_axis_off()
handles = [
    plt.Line2D([], [], color=C_RAND, lw=2.4, marker="o", ms=5, label="Random-path oracle (upper bound)"),
    plt.Line2D([], [], color=C_MCTS, lw=2.2, ls="--", marker="s", ms=4.5, label="Structured search oracle (MCTS)"),
    plt.Line2D([], [], color=C_MAJ,  lw=1.8, ls=":", marker="^", ms=4, label="Deployable vote (no judge)"),
]
axL.legend(handles=[h for h in handles], loc="upper center",
           bbox_to_anchor=(0.5, 0.98), fontsize=9.2, handlelength=2.4, labelspacing=0.9)
axL.text(0.5, 0.30,
         "Oracle = any-of-K correct\nwith a cheating selector.\n\nRandom climbs to the ceiling;\nMCTS saturates early;\nthe deployable vote stays\npinned near baseline.",
         transform=axL.transAxes, ha="center", va="top", fontsize=8.6,
         color="#4D4D4D", linespacing=1.5)

# shared axis labels
fig.supxlabel("Budget  K  (matched samples, log scale)", fontsize=11, y=0.02)
fig.supylabel("Oracle / deployable accuracy  (%)", fontsize=11, x=0.005)

fig.tight_layout(rect=[0.02, 0.03, 1, 1], w_pad=1.4, h_pad=2.0)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
base = os.path.join(HERE, "out", "fig2a_budget")
fig.savefig(base + ".svg", bbox_inches="tight")            # primary editable vector
fig.savefig(base + ".pdf", bbox_inches="tight")            # vector for LaTeX
fig.savefig(base + ".png", dpi=300, bbox_inches="tight")   # quick preview
fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight")  # submission raster
plt.close(fig)
print("saved", base + ".{svg,pdf,png,tiff}")

