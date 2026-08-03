#!/usr/bin/env python3
"""Draw R1: matched-budget routing trajectories.

Core conclusion:
Random-path reachability keeps rising after structured search saturates, while
the deployable vote remains near or below the full-model baseline.

Evidence map:
  a — hero: high-headroom ARC-Challenge case with a shared K=1 origin.
  b — generation-task validation on DART.
  c — newer-generation Qwen validation on DART.
  d — low-headroom boundary case on ARC-Challenge.

Caption facts:
  - The compared procedures use the same number of tries, scoring rule, decode
    length and default first path.
  - K=1 is identical for random-path oracle, structured-search oracle,
    deployable vote and the default full-model baseline.
  - Oracle curves require answer knowledge and are not deployable selectors.
  - Every marker is a measured budget point; no point is interpolated.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE = PACKAGE / "source-data" / "R1_routing_budget.csv"
OUTPUT = HERE / "out" / "R1_routing_budget"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    add_panel_label,
    apply_style,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.65
EXPECTED_K = [1, 2, 4, 5, 8, 16, 32, 200]
PANEL_ORDER = ["a", "b", "c", "d"]
PANEL_TITLES = {
    "a": "Llama-3.2-3B\nARC-Challenge",
    "b": "Llama-3.2-3B\nDART",
    "c": "Qwen3-4B\nDART",
    "d": "Qwen2.5-14B\nARC-Challenge",
}
Y_CONTRACT = {
    "a": ((60, 103), [60, 80, 100]),
    "b": ((-2, 53), [0, 25, 50]),
    "c": ((-2, 42), [0, 20, 40]),
    "d": ((89.5, 101), [90, 95, 100]),
}


def load_rows() -> dict[str, list[dict[str, float | int | str]]]:
    with SOURCE.open(newline="") as handle:
        raw = list(csv.DictReader(handle))

    assert len(raw) == 32, f"R1 expects 32 rows, found {len(raw)}"
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    numeric_fields = [
        "random_path_oracle",
        "structured_search_oracle",
        "deployable_vote",
        "baseline",
    ]
    for row in raw:
        converted: dict[str, float | int | str] = dict(row)
        converted["K"] = int(row["K"])
        for field in numeric_fields:
            converted[field] = float(row[field])
        grouped[row["panel"]].append(converted)

    assert set(grouped) == set(PANEL_ORDER)
    for panel in PANEL_ORDER:
        grouped[panel].sort(key=lambda row: EXPECTED_K.index(int(row["K"])))
        budgets = [int(row["K"]) for row in grouped[panel]]
        assert budgets == EXPECTED_K, f"{panel}: unexpected measured budgets {budgets}"
        origin = grouped[panel][0]
        shared = {
            float(origin["random_path_oracle"]),
            float(origin["structured_search_oracle"]),
            float(origin["deployable_vote"]),
            float(origin["baseline"]),
        }
        assert len(shared) == 1, f"{panel}: K=1 is not a shared origin"
    return grouped


def draw_panel(
    ax: plt.Axes,
    panel: str,
    rows: list[dict[str, float | int | str]],
) -> None:
    budgets = [int(row["K"]) for row in rows]
    random_oracle = [float(row["random_path_oracle"]) for row in rows]
    structured_oracle = [float(row["structured_search_oracle"]) for row in rows]
    deployable_vote = [float(row["deployable_vote"]) for row in rows]
    baseline = float(rows[0]["baseline"])

    ax.axhline(
        baseline,
        color=PALETTE["control_mid"],
        linewidth=1.0,
        linestyle=(0, (1.2, 1.8)),
        zorder=1,
    )
    ax.plot(
        budgets,
        structured_oracle,
        color=PALETTE["control_dark"],
        linewidth=1.35,
        linestyle=(0, (3.2, 2.0)),
        marker="s",
        markersize=3.0,
        markerfacecolor=PALETTE["white"],
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.plot(
        budgets,
        deployable_vote,
        color=PALETTE["realized"],
        linewidth=1.45,
        marker="D",
        markersize=3.0,
        markeredgecolor=PALETTE["white"],
        markeredgewidth=0.45,
        zorder=4,
    )
    ax.plot(
        budgets,
        random_oracle,
        color=PALETTE["reachability"],
        linewidth=1.75,
        marker="o",
        markersize=3.4,
        markeredgecolor=PALETTE["white"],
        markeredgewidth=0.45,
        zorder=5,
    )

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.78, 270)
    ax.set_xticks([1, 4, 16, 200], ["1", "4", "16", "200"])
    ax.minorticks_off()

    ylim, yticks = Y_CONTRACT[panel]
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_title(
        PANEL_TITLES[panel],
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=4.5,
        linespacing=1.0,
    )
    ax.tick_params(axis="both", which="major", length=2.7, width=0.75, pad=1.8)
    add_panel_label(
        ax,
        panel,
        x=-0.21 if panel == "a" else -0.24,
        y=1.055,
        fontsize=10.8,
    )

    if panel == "a":
        origin = float(rows[0]["random_path_oracle"])
        ax.annotate(
            "shared $K=1$",
            xy=(1, origin),
            xytext=(3.55, origin + 4.3),
            fontsize=8.0,
            color=PALETTE["control_dark"],
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "-",
                "color": PALETTE["control_mid"],
                "linewidth": 0.7,
                "shrinkA": 8,
                "shrinkB": 2,
            },
        )
        endpoint_specs = [
            (random_oracle[-1], f"{random_oracle[-1]:.0f}", PALETTE["reachability"], 1.2),
            (
                structured_oracle[-1],
                f"{structured_oracle[-1]:.1f}",
                PALETTE["control_dark"],
                0.0,
            ),
            (deployable_vote[-1], f"{deployable_vote[-1]:.0f}", PALETTE["realized"], -1.5),
        ]
        for y_value, label, color, dy in endpoint_specs:
            ax.annotate(
                label,
                xy=(200, y_value),
                xytext=(3, dy),
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                clip_on=False,
            )


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=PALETTE["reachability"],
            linewidth=1.75,
            marker="o",
            markersize=3.5,
            label="Random-path oracle",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["control_dark"],
            linewidth=1.35,
            linestyle=(0, (3.2, 2.0)),
            marker="s",
            markerfacecolor=PALETTE["white"],
            markersize=3.2,
            label="Structured-search oracle",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["realized"],
            linewidth=1.45,
            marker="D",
            markersize=3.2,
            label="Deployable vote",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["control_mid"],
            linewidth=1.0,
            linestyle=(0, (1.2, 1.8)),
            label="Full-model baseline",
        ),
    ]


def main() -> None:
    grouped = load_rows()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.28, 1.0, 1.0, 1.0],
        left=0.078,
        right=0.988,
        bottom=0.205,
        top=0.745,
        wspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    for ax, panel in zip(axes, PANEL_ORDER):
        draw_panel(ax, panel, grouped[panel])

    fig.supxlabel(r"Matched budget $K$", x=0.53, y=0.025, fontsize=9.0)
    fig.supylabel(
        "Questions correct or reachable (%)",
        x=0.012,
        y=0.475,
        fontsize=9.0,
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.45,
        columnspacing=1.0,
        fontsize=8.0,
    )

    outputs = save_publication_figure(fig, OUTPUT, dpi=600)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
