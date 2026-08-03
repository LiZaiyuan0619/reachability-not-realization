#!/usr/bin/env python3
"""Draw L1: question-level RLVR fate accounting.

Core conclusion:
Across the four clean matched K=200 cells, deployed gains come from questions
already reachable by the base, while the reachable ceiling is flat or lower.

Evidence map:
  a — hero: per-cell counts of newly produced base-reachable questions,
      newly produced base-unreachable questions, previously produced losses,
      and the resulting net deployed change.
  b — validation: paired changes in deployed accuracy and oracle reachability.

Caption facts:
  - Questions are joined by ID under the same format, temperature and K=200.
  - Base-unreachable-to-produced count is zero in all four clean cells.
  - Training also loses previously produced questions.
  - DAPO/DART shrinks reachability; SimpleRL/DART is approximately flat.
  - OR1 is excluded because its budgets/truncation are not matched.
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
SOURCE_FLOWS = PACKAGE / "source-data" / "L1_fate_flows.csv"
SOURCE_SUMMARY = PACKAGE / "source-data" / "L1_fate_summary.csv"
OUTPUT = HERE / "out" / "L1_rlvr_fate"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    apply_style,
    quiet_grid,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.65
CELL_ORDER = [
    ("SimpleRL", "dart"),
    ("SimpleRL", "gsm8k"),
    ("DAPO", "dart"),
    ("DAPO", "gsm8k"),
]
CELL_LABEL = {
    ("SimpleRL", "dart"): "SimpleRL · DART",
    ("SimpleRL", "gsm8k"): "SimpleRL · GSM8K",
    ("DAPO", "dart"): "DAPO · DART",
    ("DAPO", "gsm8k"): "DAPO · GSM8K",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[tuple[str, str], dict[str, str]],
]:
    flow_rows = read_csv(SOURCE_FLOWS)
    summary_rows = read_csv(SOURCE_SUMMARY)
    assert len(flow_rows) == 25
    assert len(summary_rows) == 4

    flows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in flow_rows:
        key = (row["pair"], row["task"])
        flows[key].append(row)
    summary = {(row["pair"], row["task"]): row for row in summary_rows}
    assert set(flows) == set(CELL_ORDER)
    assert set(summary) == set(CELL_ORDER)

    for key in CELL_ORDER:
        rows = flows[key]
        item = summary[key]
        assert {int(row["K"]) for row in rows} == {200}
        assert {int(row["n"]) for row in rows} == {300}
        assert sum(int(row["count"]) for row in rows) == 300
        assert int(item["K"]) == 200 and int(item["n"]) == 300
        gained_reachable = sum(
            int(row["count"])
            for row in rows
            if row["flow_class"] == "gained_from_base_reachable"
        )
        gained_unreachable = int(item["gained_from_base_unreachable"])
        lost = sum(
            int(row["count"])
            for row in rows
            if row["flow_class"] == "lost_from_deployed"
        )
        assert gained_reachable == int(item["gained_from_base_reachable"])
        assert gained_unreachable == 0
        assert lost == int(item["lost_from_deployed"])
        assert gained_reachable - lost == int(item["net_deployed_count"])

    assert abs(float(summary[("DAPO", "dart")]["delta_deployed_pp"]) - 14.67) < 0.01
    assert abs(float(summary[("DAPO", "dart")]["delta_oracle_pp"]) + 13.33) < 0.01
    assert abs(float(summary[("SimpleRL", "dart")]["delta_deployed_pp"]) - 12.0) < 0.01
    assert abs(float(summary[("SimpleRL", "dart")]["delta_oracle_pp"]) + 0.67) < 0.01
    return flows, summary


def y_positions() -> dict[tuple[str, str], int]:
    return {
        key: len(CELL_ORDER) - 1 - index
        for index, key in enumerate(CELL_ORDER)
    }


def signed(value: float, decimals: int = 1) -> str:
    if abs(value) < 0.05:
        return "0.0"
    return f"{value:+.{decimals}f}".replace("-", "−")


def draw_fates(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
) -> None:
    y_lookup = y_positions()
    quiet_grid(ax, axis="x")
    ax.axvline(0, color=PALETTE["control_dark"], linewidth=0.9, zorder=1)

    for key in CELL_ORDER:
        row = summary[key]
        y = y_lookup[key]
        gained = int(row["gained_from_base_reachable"])
        unreachable = int(row["gained_from_base_unreachable"])
        lost = int(row["lost_from_deployed"])
        net = int(row["net_deployed_count"])

        ax.barh(
            y,
            gained,
            height=0.40,
            left=0,
            color=PALETTE["gain"],
            alpha=0.78,
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=2,
        )
        ax.barh(
            y,
            -lost,
            height=0.40,
            left=0,
            color=PALETTE["loss"],
            alpha=0.78,
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=2,
        )
        ax.scatter(
            [0],
            [y],
            s=35,
            marker="o",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["teal"],
            linewidth=1.05,
            zorder=4,
        )
        assert unreachable == 0
        ax.scatter(
            [net],
            [y],
            s=39,
            marker="D",
            facecolor=PALETTE["reachability"],
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            f"+{gained}",
            xy=(gained, y),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["gain"],
            ha="left",
            va="center",
        )
        ax.annotate(
            f"−{lost}",
            xy=(-lost, y),
            xytext=(-4, 0),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["loss"],
            ha="right",
            va="center",
        )
        ax.annotate(
            f"net +{net}",
            xy=(net, y),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["reachability"],
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": PALETTE["white"],
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=6,
        )

    ax.set_xlim(-31, 77)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xticks([-25, 0, 25, 50, 75])
    ax.set_yticks(
        [y_lookup[key] for key in CELL_ORDER],
        [CELL_LABEL[key] for key in CELL_ORDER],
    )
    ax.set_xlabel("Question fate at $K=200$ (count; $n=300$)", fontsize=8.8)
    ax.tick_params(axis="x", length=2.7, width=0.8, pad=2)
    ax.tick_params(axis="y", length=0, labelsize=8.2, pad=4)
    ax.set_title(
        "Newly produced answers were base-reachable",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )


def draw_deltas(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
) -> None:
    y_lookup = y_positions()
    quiet_grid(ax, axis="x")
    ax.axvline(
        0,
        color=PALETTE["control_dark"],
        linewidth=0.9,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )

    for key in CELL_ORDER:
        row = summary[key]
        y = y_lookup[key]
        deployed = float(row["delta_deployed_pp"])
        oracle = float(row["delta_oracle_pp"])
        ax.plot(
            [oracle, deployed],
            [y, y],
            color=PALETTE["control_light"],
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [oracle],
            [y],
            s=35,
            marker="s",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["reachability"],
            linewidth=1.05,
            zorder=3,
        )
        ax.scatter(
            [deployed],
            [y],
            s=42,
            marker="o",
            facecolor=PALETTE["realized"],
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            signed(oracle),
            xy=(oracle, y),
            xytext=(5 if oracle < -10 else (-5 if oracle <= 0 else 5), -8),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["reachability"],
            ha="left" if oracle < -10 or oracle > 0 else "right",
            va="top",
        )
        ax.annotate(
            signed(deployed),
            xy=(deployed, y),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["realized"],
            ha="left",
            va="bottom",
        )

    ax.set_xlim(-16, 18)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xticks([-15, 0, 15])
    ax.set_yticks([y_lookup[key] for key in CELL_ORDER])
    ax.tick_params(axis="x", length=2.7, width=0.8, pad=2)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_xlabel("Change after RLVR (percentage points)", fontsize=8.8)
    ax.set_title(
        "Deployed rises; oracle does not",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )


def add_panel_labels(fig: plt.Figure) -> None:
    for label, x in [("a", 0.090), ("b", 0.665)]:
        fig.text(
            x,
            0.875,
            label,
            fontsize=10.8,
            fontweight="bold",
            color=PALETTE["ink"],
            ha="left",
            va="center",
        )


def add_legends(fig: plt.Figure) -> None:
    fate_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["gain"],
            linewidth=5.0,
            alpha=0.78,
            label="Gained: base-reachable",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["teal"],
            markersize=4.8,
            label="Gained: base-unreachable (0)",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["loss"],
            linewidth=5.0,
            alpha=0.78,
            label="Lost from deployed",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="D",
            markerfacecolor=PALETTE["reachability"],
            markeredgecolor=PALETTE["white"],
            markersize=4.8,
            label="Net deployed",
        ),
    ]
    delta_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["realized"],
            markeredgecolor=PALETTE["white"],
            markersize=5.0,
            label="$\\Delta$deployed",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["reachability"],
            markersize=4.8,
            label="$\\Delta$oracle",
        ),
    ]
    fig.legend(
        handles=fate_handles,
        loc="lower center",
        bbox_to_anchor=(0.355, 0.020),
        ncol=2,
        fontsize=8.0,
        handletextpad=0.45,
        columnspacing=0.85,
        labelspacing=0.35,
    )
    fig.legend(
        handles=delta_handles,
        loc="lower center",
        bbox_to_anchor=(0.845, 0.038),
        ncol=2,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )


def main() -> None:
    _, summary = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    ax_a = fig.add_axes([0.145, 0.290, 0.500, 0.515])
    ax_b = fig.add_axes([0.715, 0.290, 0.255, 0.515])

    draw_fates(ax_a, summary)
    draw_deltas(ax_b, summary)
    add_panel_labels(fig)
    add_legends(fig)

    outputs = save_publication_figure(fig, OUTPUT)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
