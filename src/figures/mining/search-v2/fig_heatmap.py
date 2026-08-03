#!/usr/bin/env python3
"""Fig 2b: universality heatmap. Delta = random@64 oracle - MCTS ceiling.
All cells positive -> sequential colormap (no diverging washout).
Rows=models, cols=tasks, both sorted by mean gap. format_collapse excluded
(main text); empty model x task combos shown as grey 'n/a'."""
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

HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "..", "v1", "data", "heatmap_46cells.csv")
rows = [r for r in csv.DictReader((l for l in open(SRC) if not l.startswith("#")))]

# EXCLUSION (recorded): drop format_collapse cells for main text
n_before = len(rows)
rows = [r for r in rows if r["flag"] == "ok"]
n_after = len(rows)
print(f"exclusion: {n_before} -> {n_after} cells (dropped {n_before-n_after} format_collapse)")

MODEL_ORDER = ["Llama-3.2-3B", "Llama-3.1-8B", "Mistral-7B", "Qwen3-4B-2507", "Qwen2.5-3B", "Qwen2.5-14B"]
MODEL_DISP  = {"Llama-3.2-3B":"Llama-3.2-3B","Llama-3.1-8B":"Llama-3.1-8B","Mistral-7B":"Mistral-7B",
               "Qwen3-4B-2507":"Qwen3-4B","Qwen2.5-3B":"Qwen2.5-3B","Qwen2.5-14B":"Qwen2.5-14B"}
TASK_DISP = {"arc_challenge":"ARC-c","arc_easy":"ARC-e","dart":"DART","dart_pool":"DART-p",
             "mmlu":"MMLU","hellaswag":"HellaS","winogrande":"WinoG","piqa":"PIQA",
             "boolq":"BoolQ","openbookqa":"OBQA","commonsense_qa":"CSQA"}

# build matrix
tasks = list(TASK_DISP.keys())
M = np.full((len(MODEL_ORDER), len(tasks)), np.nan)
for r in rows:
    if r["model"] in MODEL_ORDER and r["task"] in tasks:
        M[MODEL_ORDER.index(r["model"]), tasks.index(r["task"])] = float(r["rand_minus_mcts"])

# sort cols by column-mean gap (desc), rows by row-mean gap (desc)
col_order = np.argsort(-np.nanmean(M, axis=0))
M = M[:, col_order]; tasks = [tasks[i] for i in col_order]
row_order = np.argsort(-np.nanmean(M, axis=1))
M = M[row_order, :]; models = [MODEL_ORDER[i] for i in row_order]
print("gap range:", np.nanmin(M), "->", np.nanmax(M), "| n cells:", np.sum(~np.isnan(M)))

# ── render ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.4))
cmap = plt.get_cmap("YlGnBu").copy()
cmap.set_bad("#EDEDED")            # empty model x task -> light grey
vmax = np.ceil(np.nanmax(M) / 5) * 5
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

ax.set_xticks(range(len(tasks))); ax.set_xticklabels([TASK_DISP[t] for t in tasks], fontsize=9.5)
ax.set_yticks(range(len(models))); ax.set_yticklabels([MODEL_DISP[m] for m in models], fontsize=9.5)
ax.tick_params(length=0)
ax.set_frame_on(False)
# minor-grid white separators for a crisp matrix
ax.set_xticks(np.arange(-.5, len(tasks), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.2)
ax.tick_params(which="minor", length=0)

# annotate every cell
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#B0B0B0")
        else:
            r, g, b, _ = cmap(norm(v))
            lum = 0.299*r + 0.587*g + 0.114*b
            ax.text(j, i, f"+{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if lum < 0.5 else "#1A1A1A")

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Random-path oracle  −  MCTS ceiling   (pp)", fontsize=9.5)
cbar.outline.set_visible(False)
ax.set_title(f"Random path $\\geq$ structured search in all {np.sum(~np.isnan(M)):.0f} cells "
             "(every cell positive)",
             fontsize=11, pad=8)

fig.tight_layout()
base = os.path.join(HERE, "out", "fig2b_heatmap")
fig.savefig(base + ".svg", bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved", base + ".{svg,pdf,png}")
