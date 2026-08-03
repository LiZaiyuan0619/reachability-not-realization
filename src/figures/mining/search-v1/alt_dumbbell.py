"""Alt encoding — per-cell dumbbell: MCTS ceiling ->  random ceiling.

Each row a cell; a line from MCTS (blue) to random (red) shows the gap length.
All arrows point right (random higher). Sorted by gap. A clean 'lift' visual,
alternative to the heatmap for the same B.1.1 data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "heatmap_46cells.csv")

TOP_N = 46   # show all; set e.g. 24 to declutter


def main():
    S.apply_rc()
    df = pd.read_csv(DATA, comment="#").dropna(subset=["rand_minus_mcts"])
    df = df.sort_values("rand_minus_mcts").tail(TOP_N).reset_index(drop=True)
    df["label"] = (df["model"].str.replace("Llama-", "L").str.replace("Qwen", "Q")
                   + "/" + df["task"])
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.6, 8.4))
    ax.hlines(y, df["mcts"], df["rand64"], color="#CCC", lw=1.4, zorder=1)
    ax.scatter(df["mcts"], y, s=22, color=S.C_MCTS, zorder=3, label="MCTS ceiling")
    ax.scatter(df["rand64"], y, s=22, color=S.C_RANDOM, zorder=3, label="random ceiling")
    ax.set_yticks(y); ax.set_yticklabels(df["label"], fontsize=S.FS_TICK-1.5)
    ax.set_xlabel("oracle ceiling (%)")
    ax.set_xlim(0, 105)
    ax.set_title("random lifts the ceiling over search in all 46 cells", fontsize=S.FS_LABEL)
    ax.grid(axis="x", alpha=S.GRID_A, lw=0.5)
    ax.legend(loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"alt_dumbbell.{ext}"))
    print("[alt_dumbbell] saved png+pdf")


if __name__ == "__main__":
    main()
