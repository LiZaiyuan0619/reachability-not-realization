#!/usr/bin/env python3
"""Fig 2c: sorted dot-strip. Every one of 43 cells has random-oracle > MCTS,
so all points sit above the zero line -> 'no exception' at a glance.
Colored by model family (Llama blue / Qwen red / Mistral teal)."""
import os, csv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 10, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 1.0, "legend.frameon": False,
})

FAM_COLOR = {"Llama": "#0F4D92", "Qwen": "#B64342", "Mistral": "#42949E"}
def fam_of(m):
    if m.startswith("Llama"): return "Llama"
    if m.startswith("Qwen"):  return "Qwen"
    return "Mistral"

HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "..", "v1", "data", "heatmap_46cells.csv")
rows = [r for r in csv.DictReader((l for l in open(SRC) if not l.startswith("#")))]
n_before = len(rows)
rows = [r for r in rows if r["flag"] == "ok"]            # exclude format_collapse
print(f"exclusion: {n_before} -> {len(rows)} cells")

data = sorted([(float(r["rand_minus_mcts"]), fam_of(r["model"]), r["model"], r["task"]) for r in rows])
gaps = np.array([d[0] for d in data])
cols = [FAM_COLOR[d[1]] for d in data]
x = np.arange(len(data))

fig, ax = plt.subplots(figsize=(11, 4.2))
# stems from zero to each point
ax.vlines(x, 0, gaps, color=cols, lw=1.4, alpha=0.55, zorder=1)
ax.scatter(x, gaps, c=cols, s=46, edgecolors="white", linewidths=0.8, zorder=3)
ax.axhline(0, color="#272727", lw=1.4, zorder=2)

# annotate min and max cells
imin, imax = int(np.argmin(gaps)), int(np.argmax(gaps))
for i, va, dy in [(imin, "bottom", 1.5), (imax, "top", -1.5)]:
    ax.annotate(f"{data[i][2].replace('Qwen','Q').replace('Llama-','L')}/{data[i][3][:5]}\n+{gaps[i]:.1f}",
                (x[i], gaps[i]), textcoords="offset points", xytext=(0, 12 if va=="bottom" else -28),
                ha="center", va=va, fontsize=7.5, color="#4D4D4D")

ax.set_ylim(-3, np.max(gaps) + 8)
ax.set_xlim(-1, len(data))
ax.set_xticks([])
ax.set_ylabel("Random-path oracle  −  MCTS ceiling   (pp)", fontsize=10)
ax.set_xlabel(f"{len(data)} model $\\times$ benchmark cells, sorted by gap", fontsize=10)
ax.text(0.02, 0.94, "All 43 cells lie above zero — no exception",
        transform=ax.transAxes, fontsize=10.5, color="#272727", fontweight="bold")

handles = [plt.Line2D([], [], marker="o", ls="", ms=8, mec="white",
                      color=c, label=f) for f, c in FAM_COLOR.items()]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.86),
          fontsize=9.2, ncol=3, columnspacing=1.2, handletextpad=0.3)

fig.tight_layout()
base = os.path.join(HERE, "out", "fig2c_strip")
fig.savefig(base + ".svg", bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved", base + ".{svg,pdf,png}")
