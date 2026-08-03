"""Panel (d): action-space anti-attack. random wins in drllm 46/46, POLAR 23/23, SKIPONLY 10/10.

Two readable encodings tried (v1 broad-net):
  variant A: 'win count' bars with N annotated + mean-gap as bar height on twin axis
  variant B: dumbbell of gap range per space, with cells-count label
This file draws variant A (win-rate = 100% everywhere) with mean gap overlaid.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "action_spaces.csv")


def load():
    return pd.read_csv(DATA, comment="#")


def draw(ax, df):
    x = np.arange(len(df))
    # mean gap as bar height (the substance); annotate the k/k win count
    bars = ax.bar(x, df["mean_gap"], width=0.55, color=S.C_RANDOM, alpha=0.85, zorder=3)
    # gap range whiskers
    ax.errorbar(x, df["mean_gap"],
                yerr=[df["mean_gap"] - df["gap_lo"], df["gap_hi"] - df["mean_gap"]],
                fmt="none", ecolor="#333", elinewidth=1, capsize=4, zorder=4)
    for xi, r in df.iterrows():
        ax.text(xi, r["gap_hi"] + 2, f"{int(r['cells_random_wins'])}/{int(r['cells_total'])}",
                ha="center", fontsize=S.FS_ANNOT+0.5, fontweight="bold", color="#222")
        ax.text(xi, -4.5, r["note"], ha="center", fontsize=S.FS_ANNOT-1,
                color="#666", wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels(df["action_space"], fontsize=S.FS_LABEL, fontweight="bold")
    ax.set_ylabel("random − MCTS gap (pp)\nmean ± range")
    ax.set_ylim(0, df["gap_hi"].max() + 10)
    ax.set_title("random wins in every action space", fontsize=S.FS_LABEL)
    ax.grid(axis="y", alpha=S.GRID_A, lw=0.5)


if __name__ == "__main__":
    S.apply_rc()
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    draw(ax, load())
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"p_spaces.{ext}"))
    print("[p_spaces] saved png+pdf")
