"""Panel (b): budget curve. MCTS saturates ~76.7, random climbs to 100 (l3b/ARC).

NOTE: intermediate K points are interpolated for shape; anchors (K=1/64/200,
MCTS saturation, sampling) are real (appendix B.1.1/B.1.4). Replace with true
per-rank values when available.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "budget_curve.csv")

# ---- panel-local knobs ----
ANNOT_SAT = True     # annotate MCTS saturation ceiling
FILL_GAP  = True     # shade the oracle gap between random and MCTS


def load():
    return pd.read_csv(DATA, comment="#")


def draw(ax, df):
    K = df["K"]
    ax.plot(K, df["random_oracle"], "-o", color=S.C_RANDOM, lw=S.LW_CURVE,
            ms=S.MS_MARKER, label="random path (oracle)", zorder=5)
    ax.plot(K, df["mcts_ceiling"], "-s", color=S.C_MCTS, lw=S.LW_CURVE,
            ms=S.MS_MARKER, label="MCTS structured search", zorder=4)
    ax.plot(K, df["sampling_oracle"], "--^", color=S.C_SAMPLING, lw=1.6,
            ms=S.MS_MARKER-1, label="sampling (oracle)", zorder=3)

    if FILL_GAP:
        ax.fill_between(K, df["mcts_ceiling"], df["random_oracle"],
                        color=S.C_RANDOM, alpha=0.10, zorder=1)

    ax.set_xscale("log", base=2)
    ax.set_xticks(K)
    ax.set_xticklabels([int(k) for k in K])
    ax.set_xlabel("budget K (tries, log scale)")
    ax.set_ylabel("oracle accuracy (%)")
    ax.set_ylim(60, 103)
    ax.set_title("l3b / ARC-challenge: search saturates, random unlocks", fontsize=S.FS_LABEL)
    ax.grid(alpha=S.GRID_A, lw=0.5)

    if ANNOT_SAT:
        ax.annotate("MCTS ceiling 76.7\n(saturates ~rank 9)",
                    xy=(200, 76.7), xytext=(24, 68),
                    fontsize=S.FS_ANNOT, color=S.C_MCTS,
                    arrowprops=dict(arrowstyle="->", color=S.C_MCTS, lw=0.8))
        ax.annotate("random → 100",
                    xy=(200, 100), xytext=(40, 101),
                    fontsize=S.FS_ANNOT, color=S.C_RANDOM)
    ax.legend(loc="lower right")


if __name__ == "__main__":
    S.apply_rc()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    draw(ax, load())
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"p_budget.{ext}"))
    print("[p_budget] saved png+pdf")
