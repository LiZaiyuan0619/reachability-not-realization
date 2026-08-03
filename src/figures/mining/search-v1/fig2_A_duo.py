"""Candidate A — heatmap hero + budget curve, 2 panels.

Layout (per user rule: 2 subplots => vertical stack):
  (a) heatmap  (top, hero)
  (b) budget curve (bottom)
Search-family / action-space evidence cited to appendix (not drawn here).
Leaner alternative to candidate B.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import _style as S
import p_heatmap, p_budget

HERE = os.path.dirname(os.path.abspath(__file__))

FIGSIZE = (6.2, 9.0)      # single-column tall
H_RATIOS = [1.5, 1.0]
HSPACE = 0.28


def main():
    S.apply_rc()
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 1, height_ratios=H_RATIOS, hspace=HSPACE, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    p_heatmap.draw(ax_a, p_heatmap.load())
    p_budget.draw(ax_b, p_budget.load())
    S.panel_letter(ax_a, "a", dx=-0.12)
    S.panel_letter(ax_b, "b", dx=-0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig2_A_duo.{ext}"))
    print("[fig2_A_duo] saved png+pdf")


if __name__ == "__main__":
    main()
