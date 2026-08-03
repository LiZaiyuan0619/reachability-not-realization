#!/usr/bin/env python3
"""Draw R2: routing breadth and tested robustness.

Core conclusion:
The random-path oracle advantage is positive in every clean audited model-task
cell and survives the tested changes in action space and search family.

Evidence map:
  a — hero: all 43 clean model-task effects, sorted by advantage.
  b — validation: exact action-space win counts and cell-level advantages over
      hill climbing and genetic search.

Caption facts:
  - 46 model-task cells were evaluated; 43 form the clean headline and three
    Qwen2.5-3B format-collapse evaluations are flagged and excluded.
  - All 43 clean random-minus-structured effects are positive.
  - Action-space rows report exact directional counts, not approximate means.
  - Search-family dots are the 12 measured cells for each comparison at K=64.
  - The result concerns only the tested routing and search procedures.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE_CELLS = PACKAGE / "source-data" / "R2_routing_cells.csv"
SOURCE_SPACES = PACKAGE / "source-data" / "R2_action_spaces.csv"
SOURCE_FAMILIES = PACKAGE / "source-data" / "R2_search_families.csv"
OUTPUT = HERE / "out" / "R2_routing_breadth"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    add_panel_label,
    apply_style,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.55
FAMILY_STYLE = {
    "Llama": {"color": PALETTE["reachability"], "marker": "o"},
    "Mistral": {"color": PALETTE["teal"], "marker": "s"},
    "Qwen": {"color": PALETTE["reachability_mid"], "marker": "D"},
}
ACTION_LABELS = {
    "drllm": "Native",
    "POLAR": "POLAR",
    "SKIPONLY": "SKIPONLY",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, list[float]],
]:
    cells = read_csv(SOURCE_CELLS)
    assert len(cells) == 46
    clean = [row for row in cells if row["include_clean_headline"] == "1"]
    flagged = [row for row in cells if row["include_clean_headline"] == "0"]
    assert len(clean) == 43
    assert len(flagged) == 3
    assert {row["analysis_status"] for row in flagged} == {"format_collapse"}
    assert all(float(row["random_minus_structured_pp"]) > 0 for row in clean)
    clean.sort(key=lambda row: int(row["clean_rank"]))
    assert [int(row["clean_rank"]) for row in clean] == list(range(1, 44))

    spaces = read_csv(SOURCE_SPACES)
    assert len(spaces) == 3
    expected_counts = {"drllm": (46, 46), "POLAR": (23, 23), "SKIPONLY": (10, 10)}
    for row in spaces:
        pair = (int(row["cells_total"]), int(row["cells_random_wins"]))
        assert pair == expected_counts[row["action_space"]]
        assert float(row["win_fraction"]) == 1.0

    family_rows = read_csv(SOURCE_FAMILIES)
    assert len(family_rows) == 36
    by_cell: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in family_rows:
        key = (row["model"], row["task"], row["seed"])
        by_cell[key][row["method"]] = float(row["oracle_at_64"])
    assert len(by_cell) == 12
    advantages: dict[str, list[float]] = {
        "Hill climbing": [],
        "Genetic search": [],
    }
    for methods in by_cell.values():
        assert set(methods) == {"random path", "hill climbing", "genetic search"}
        random_value = methods["random path"]
        hill_gap = random_value - methods["hill climbing"]
        genetic_gap = random_value - methods["genetic search"]
        assert hill_gap >= 0 and genetic_gap >= 0
        advantages["Hill climbing"].append(hill_gap)
        advantages["Genetic search"].append(genetic_gap)
    return clean, spaces, advantages


def draw_breadth(ax: plt.Axes, clean: list[dict[str, str]]) -> None:
    ranks = [int(row["clean_rank"]) for row in clean]
    effects = [float(row["random_minus_structured_pp"]) for row in clean]

    ax.vlines(
        ranks,
        0,
        effects,
        color=PALETTE["control_light"],
        linewidth=0.7,
        zorder=1,
    )
    for family, style in FAMILY_STYLE.items():
        rows = [row for row in clean if row["family"] == family]
        ax.scatter(
            [int(row["clean_rank"]) for row in rows],
            [float(row["random_minus_structured_pp"]) for row in rows],
            s=22,
            marker=style["marker"],
            color=style["color"],
            edgecolor=PALETTE["white"],
            linewidth=0.45,
            zorder=3,
        )

    ax.axhline(
        0,
        color=PALETTE["control_dark"],
        linewidth=0.95,
        linestyle=(0, (3.0, 2.0)),
        zorder=2,
    )
    ax.set_xlim(0, 44)
    ax.set_ylim(-1.0, 41.0)
    ax.set_xticks([1, 10, 20, 30, 43])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Clean cell rank (sorted by advantage)", fontsize=9.0)
    ax.set_ylabel("Random − structured oracle (%)", fontsize=9.0)
    ax.tick_params(axis="both", length=2.8, width=0.8, pad=2)
    ax.set_title(
        "Random advantage in every clean cell",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    ax.text(
        1.0,
        1.035,
        "46 evaluated · 43 clean · 3 flagged",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=PALETTE["control_dark"],
    )
    ax.text(
        0.025,
        0.91,
        "43/43 > 0",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=PALETTE["reachability"],
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": PALETTE["band_aqua"],
            "edgecolor": "none",
        },
    )
    add_panel_label(ax, "a", x=-0.105, y=1.045, fontsize=10.8)

    minimum = clean[0]
    maximum = clean[-1]
    ax.annotate(
        f"min {float(minimum['random_minus_structured_pp']):.1f}%\n"
        "Qwen2.5-14B · BoolQ",
        xy=(int(minimum["clean_rank"]), float(minimum["random_minus_structured_pp"])),
        xytext=(2.8, 20.5),
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["control_mid"],
            "linewidth": 0.75,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )
    ax.annotate(
        f"max {float(maximum['random_minus_structured_pp']):.1f}%\n"
        "Llama-3.2-3B · HellaSwag",
        xy=(int(maximum["clean_rank"]), float(maximum["random_minus_structured_pp"])),
        xytext=(33.0, 36.0),
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="right",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["control_mid"],
            "linewidth": 0.75,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def draw_action_space_table(ax: plt.Axes, spaces: list[dict[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Persists across tested variations",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=4,
    )
    add_panel_label(ax, "b", x=-0.18, y=1.06, fontsize=10.8)

    ax.text(
        0.0,
        0.84,
        "Action space",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
        color=PALETTE["control_dark"],
    )
    ax.text(
        1.0,
        0.84,
        "Random wins",
        fontsize=8.0,
        fontweight="bold",
        ha="right",
        va="center",
        color=PALETTE["control_dark"],
    )
    y_positions = [0.60, 0.36, 0.12]
    ordered = sorted(spaces, key=lambda row: ["drllm", "POLAR", "SKIPONLY"].index(row["action_space"]))
    for row, y in zip(ordered, y_positions):
        ax.scatter(
            [0.025],
            [y],
            s=22,
            marker="s",
            color=PALETTE["reachability"],
            edgecolor=PALETTE["white"],
            linewidth=0.45,
            clip_on=False,
        )
        ax.text(
            0.09,
            y,
            ACTION_LABELS[row["action_space"]],
            fontsize=8.2,
            ha="left",
            va="center",
        )
        ax.text(
            1.0,
            y,
            f"{row['cells_random_wins']} / {row['cells_total']}",
            fontsize=8.4,
            fontweight="bold",
            color=PALETTE["reachability"],
            ha="right",
            va="center",
        )
        ax.plot(
            [0.08, 1.0],
            [y - 0.11, y - 0.11],
            color=PALETTE["grid"],
            linewidth=0.6,
            clip_on=False,
        )


def draw_search_family(
    ax: plt.Axes,
    advantages: dict[str, list[float]],
) -> None:
    labels = ["Genetic search", "Hill climbing"]
    display_labels = ["Genetic\nsearch", "Hill\nclimbing"]
    y_positions = {"Genetic search": 0, "Hill climbing": 1}
    jitter = [-0.16, -0.11, -0.07, -0.03, 0.01, 0.05, 0.09, 0.13, -0.13, -0.05, 0.07, 0.15]

    ax.axvline(
        0,
        color=PALETTE["control_dark"],
        linewidth=0.9,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )
    for label in labels:
        values = advantages[label]
        y = y_positions[label]
        ax.scatter(
            values,
            [y + offset for offset in jitter],
            s=18,
            color=PALETTE["reachability_soft"],
            edgecolor=PALETTE["reachability_mid"],
            linewidth=0.45,
            zorder=2,
        )
        median = statistics.median(values)
        ax.scatter(
            [median],
            [y],
            s=34,
            marker="D",
            color=PALETTE["reachability"],
            edgecolor=PALETTE["white"],
            linewidth=0.55,
            zorder=4,
        )
        label_on_left = median > 15.0
        ax.text(
            median - 1.2 if label_on_left else median + 1.1,
            y + 0.18,
            f"median {median:.1f}",
            fontsize=9.2,
            color=PALETTE["reachability"],
            fontweight="bold",
            ha="right" if label_on_left else "left",
            va="center",
        )

    ax.set_xlim(-1.0, 31.0)
    ax.set_ylim(-0.43, 1.43)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 1], display_labels)
    ax.set_xlabel("Random − search oracle (%)", fontsize=8.8)
    ax.tick_params(axis="both", length=2.6, width=0.75, pad=2)
    ax.set_title(
        "Search-family control\n(12 measured cells each)",
        loc="left",
        fontsize=8.0,
        fontweight="bold",
        pad=3,
    )


def family_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=FAMILY_STYLE[family]["marker"],
            markerfacecolor=FAMILY_STYLE[family]["color"],
            markeredgecolor=PALETTE["white"],
            markersize=5.0,
            label=family,
        )
        for family in ["Llama", "Mistral", "Qwen"]
    ]


def main() -> None:
    clean, spaces, advantages = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.55, 1.0],
        left=0.085,
        right=0.985,
        bottom=0.21,
        top=0.79,
        wspace=0.20,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(
        2,
        1,
        height_ratios=[0.88, 1.12],
        hspace=0.62,
    )
    ax_b_top = fig.add_subplot(right[0, 0])
    ax_b_bottom = fig.add_subplot(right[1, 0])

    draw_breadth(ax_a, clean)
    draw_action_space_table(ax_b_top, spaces)
    draw_search_family(ax_b_bottom, advantages)
    fig.legend(
        handles=family_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.34, 0.98),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=8.0,
    )

    outputs = save_publication_figure(fig, OUTPUT, dpi=600)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
