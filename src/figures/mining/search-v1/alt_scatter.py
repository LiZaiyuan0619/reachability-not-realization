"""Alt encoding — MCTS ceiling vs random ceiling scatter (all 46 cells).

Every cell above the y=x line => random >= structured search. A different way to
show universality: a cloud entirely on one side of the diagonal. Point size ~ -log10(p),
color by model family. Good for reviewers who distrust heatmap color scales.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "heatmap_46cells.csv")

FAM_COLOR = {
    "Llama-3.2-3B": "#D1495B", "Llama-3.1-8B": "#E8846B", "Mistral-7B": "#EDAE49",
    "Qwen2.5-3B": "#30638E", "Qwen2.5-14B": "#5E86AD", "Qwen3-4B-2507": "#00798C",
}


def main():
    S.apply_rc()
    df = pd.read_csv(DATA, comment="#")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 100], [0, 100], "--", color="#888", lw=1, zorder=1, label="y = x (tie)")
    for m, g in df.groupby("model"):
        sz = 12 + 2.2 * g["neg_log10_p"].clip(upper=60)
        mk = "v" if (g["flag"] == "format_collapse").any() else "o"
        ax.scatter(g["mcts"], g["rand64"], s=sz, c=FAM_COLOR.get(m, "#555"),
                   edgecolor="white", lw=0.5, alpha=0.9, label=m, zorder=3)
    ax.set_xlabel("MCTS structured-search ceiling (%)")
    ax.set_ylabel("random-path oracle ceiling (%)")
    ax.set_xlim(0, 102); ax.set_ylim(0, 105)
    ax.set_title("every cell above the diagonal: random ≥ search (46/46)",
                 fontsize=S.FS_LABEL)
    ax.grid(alpha=S.GRID_A, lw=0.5)
    ax.legend(loc="lower right", fontsize=S.FS_ANNOT-0.5, ncol=1)
    ax.text(0.03, 0.95, "point size ∝ −log₁₀ p\n▼ = format-collapse cell",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, va="top", color="#555")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"alt_scatter.{ext}"))
    print("[alt_scatter] saved png+pdf")


if __name__ == "__main__":
    main()
