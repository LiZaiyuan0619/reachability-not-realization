"""Panel (a): 46-cell rand-minus-MCTS heatmap. Standalone + importable.

HERO panel. Rows=task, Cols=model. Warm = random beats structured search.
format_collapse cells marked with a dot (shown but not a headline).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "heatmap_46cells.csv")

# ---- panel-local knobs ----
MODEL_ORDER = ["Llama-3.2-3B", "Llama-3.1-8B", "Mistral-7B",
               "Qwen2.5-3B", "Qwen2.5-14B", "Qwen3-4B-2507"]
TASK_ORDER = ["dart_pool", "dart", "hellaswag", "mmlu", "arc_challenge",
              "winogrande", "openbookqa", "commonsense_qa", "piqa",
              "boolq", "arc_easy"]
CELL_FONT = 6.5


def load():
    df = pd.read_csv(DATA, comment="#")
    return df


def draw(ax, df, cbar=True):
    piv = df.pivot(index="task", columns="model", values="rand_minus_mcts")
    piv = piv.reindex(index=TASK_ORDER, columns=MODEL_ORDER)
    flag = df.pivot(index="task", columns="model", values="flag").reindex(
        index=TASK_ORDER, columns=MODEL_ORDER)

    im = ax.imshow(piv.values, cmap=S.CMAP_GAP, vmin=S.HEAT_VMIN,
                   vmax=S.HEAT_VMAX, aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right", fontsize=S.FS_TICK)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER, fontsize=S.FS_TICK)
    ax.set_title("random − structured-search  (oracle ceiling, pp)", fontsize=S.FS_LABEL)

    for i in range(len(TASK_ORDER)):
        for j in range(len(MODEL_ORDER)):
            v = piv.values[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#F2F2F2",
                                       edgecolor="white"))
                continue
            if S.HEAT_TEXT:
                tc = "white" if v > 32 else "#222222"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=CELL_FONT, color=tc)
            if flag.values[i, j] == "format_collapse":
                ax.plot(j, i-0.34, marker="v", ms=3.5, color=S.C_FLAG)

    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TASK_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("gap (pp)", fontsize=S.FS_ANNOT)
        cb.ax.tick_params(labelsize=S.FS_ANNOT)
    return im


if __name__ == "__main__":
    S.apply_rc()
    df = load()
    n_cells = df.dropna(subset=["rand_minus_mcts"]).shape[0]
    n_pos = (df["rand_minus_mcts"] > 0).sum()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    draw(ax, df)
    ax.text(0.5, 1.13, f"random ≥ structured search in {n_pos}/{n_cells} cells (all positive)",
            transform=ax.transAxes, ha="center", fontsize=S.FS_LABEL, fontweight="bold")
    fig.text(0.5, -0.02, "▼ = greedy format-collapse cell (lenient-scored, not a headline)",
             ha="center", fontsize=S.FS_ANNOT, color="#555")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"p_heatmap.{ext}"))
    print(f"[p_heatmap] {n_pos}/{n_cells} positive; saved png+pdf")
