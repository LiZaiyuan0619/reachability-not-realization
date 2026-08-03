"""Candidate B — 'three-panel defense wall' + bottom budget curve.

Layout (per user rule: 3 subplots => horizontal row):
  top row (3 horizontal panels):  (a) heatmap | (c) families | (d) spaces
  bottom (full width):            (b) budget curve
The heatmap is the hero (widest). Reviewer risks 1&2 answered by (c)(d) in-body.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import _style as S
import p_heatmap, p_budget, p_families, p_spaces

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- figure-level knobs ----
FIGSIZE = (13.0, 8.2)     # AAAI double-col wide; tune down for final
W_RATIOS = [1.5, 1.25, 1.0]   # heatmap hero widest
H_RATIOS = [1.35, 1.0]
WSPACE, HSPACE = 0.42, 0.32


def main():
    S.apply_rc()
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 3, width_ratios=W_RATIOS, height_ratios=H_RATIOS,
                  wspace=WSPACE, hspace=HSPACE, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[1, :])

    p_heatmap.draw(ax_a, p_heatmap.load())
    p_families.draw(ax_c, p_families.load())
    p_spaces.draw(ax_d, p_spaces.load())
    p_budget.draw(ax_b, p_budget.load())

    for ax, L in [(ax_a, "a"), (ax_c, "c"), (ax_d, "d"), (ax_b, "b")]:
        S.panel_letter(ax, L)

    fig.suptitle("Matched-budget random layer paths reach ceilings at or above every tested search",
                 fontsize=S.FS_TITLE+1, fontweight="bold", y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig2_B_defensewall.{ext}"))
    print("[fig2_B_defensewall] saved png+pdf")


if __name__ == "__main__":
    main()
