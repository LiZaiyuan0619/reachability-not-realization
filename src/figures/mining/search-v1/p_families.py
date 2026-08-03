"""Panel (c): search-family anti-attack. rand >= ga >= hc, 12/12 cells.

Grouped bars per model/task cell: random vs genetic vs hillclimb oracle ceiling,
with the deploy baseline as a reference line. Shows random tops all three families.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "search_families.csv")

# ---- panel-local knobs ----
SORT_BY = "rand_O"   # order cells by random ceiling


def load():
    df = pd.read_csv(DATA, comment="#")
    df["cell"] = df["model"].str.replace("Llama-", "L").str.replace("Qwen", "Q") \
        + "/" + df["task"] + "·" + df["seed"].astype(str)
    return df.sort_values(SORT_BY, ascending=True).reset_index(drop=True)


def draw(ax, df):
    y = np.arange(len(df))
    h = S.BAR_W
    ax.barh(y + h, df["rand_O"], height=h, color=S.C_RANDOM, label="random", zorder=3)
    ax.barh(y,      df["ga_O"],  height=h, color=S.C_GA, label="genetic", zorder=3)
    ax.barh(y - h, df["hc_O"],   height=h, color=S.C_HC, label="hillclimb", zorder=3)
    # baseline ticks
    ax.plot(df["orig"], y, "|", color=S.C_BASE, ms=9, mew=1.5, label="deploy baseline", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["cell"], fontsize=S.FS_TICK-0.5)
    ax.set_xlabel("oracle ceiling reached (%)")
    ax.set_xlim(0, 105)
    ax.set_title("random tops every search family (12/12)", fontsize=S.FS_LABEL)
    ax.grid(axis="x", alpha=S.GRID_A, lw=0.5)
    ax.legend(loc="lower right", ncol=2)


if __name__ == "__main__":
    S.apply_rc()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    draw(ax, load())
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"p_families.{ext}"))
    print("[p_families] saved png+pdf")
