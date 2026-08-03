#!/usr/bin/env python3
"""Draw R4: answer-blind selector recovery and temperature sensitivity.

Core conclusion:
The tested answer-blind selectors recover little or none of the
oracle-deployment gap, while higher temperature widens that gap.

Evidence map:
  a — hero: cell-level recovery distributions and clean medians at K=32.
  b — validation: measured oracle-minus-majority gaps across temperature.

Caption facts:
  - Recovery is (selector - greedy) / (oracle - greedy), shown as percent.
  - Panel a uses a common K=32 comparison.
  - Three Qwen2.5-3B format-collapse ratios are hollow diagnostic points and
    are excluded from the clean medians.
  - Panel b shows measured temperatures only; 8B/HellaSwag has no t=0.7 point.
  - Temperature does not provide the selector with answer knowledge.
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
SOURCE_RECOVERY = PACKAGE / "source-data" / "R4_selector_recovery.csv"
SOURCE_SUMMARY = PACKAGE / "source-data" / "R4_selector_summary.csv"
SOURCE_TEMPERATURE = PACKAGE / "source-data" / "R4_temperature.csv"
OUTPUT = HERE / "out" / "R4_routing_selection"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    add_panel_label,
    apply_style,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.60
CHANNEL_ORDER = [
    "majority vote",
    "confidence",
    "MCTS vote",
    "hill climbing",
    "genetic search",
    "self-verification",
]
CHANNEL_LABELS = {
    "majority vote": "Majority\nvote",
    "confidence": "Confidence",
    "MCTS vote": "MCTS\nvote",
    "hill climbing": "Hill\nclimbing",
    "genetic search": "Genetic\nsearch",
    "self-verification": "Self-\nverification",
}
SERIES_STYLE = {
    "3B/DART": {
        "color": PALETTE["reachability"],
        "marker": "o",
        "linestyle": "-",
        "label": "3B DART",
        "label_offset": 1.3,
    },
    "3B/DART-pool": {
        "color": PALETTE["reachability_mid"],
        "marker": "s",
        "linestyle": (0, (3.0, 1.8)),
        "label": "3B DART-pool",
        "label_offset": 0.5,
    },
    "8B/ARC-ch": {
        "color": PALETTE["teal"],
        "marker": "^",
        "linestyle": "-",
        "label": "8B ARC-c",
        "label_offset": 0.0,
    },
    "8B/HellaSwag": {
        "color": PALETTE["control_dark"],
        "marker": "D",
        "linestyle": (0, (2.2, 1.6)),
        "label": "8B Hella",
        "label_offset": -1.6,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, float],
    dict[str, list[tuple[float, float]]],
]:
    recovery = read_csv(SOURCE_RECOVERY)
    summary = read_csv(SOURCE_SUMMARY)
    assert len(recovery) == 78
    assert len(summary) == 6

    by_channel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in recovery:
        by_channel[row["channel"]].append(row)
    assert set(by_channel) == set(CHANNEL_ORDER)

    summary_lookup = {
        row["channel"]: float(row["clean_median_recovery_percent"]) for row in summary
    }
    flagged_total = 0
    for channel in CHANNEL_ORDER:
        rows = by_channel[channel]
        clean = [
            float(row["recovery_percent"])
            for row in rows
            if row["include_clean_median"] == "1"
        ]
        flagged = [row for row in rows if row["include_clean_median"] == "0"]
        flagged_total += len(flagged)
        assert abs(statistics.median(clean) - summary_lookup[channel]) < 0.02
        assert all(row["analysis_status"] == "format_collapse" for row in flagged)
    assert flagged_total == 3

    temperature_rows = read_csv(SOURCE_TEMPERATURE)
    assert len(temperature_rows) == 11
    by_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in temperature_rows:
        by_series[row["cell"]].append(
            (float(row["temperature"]), float(row["selection_gap_pp"]))
        )
    assert set(by_series) == set(SERIES_STYLE)
    for points in by_series.values():
        points.sort()
    assert [x for x, _ in by_series["8B/HellaSwag"]] == [1.0, 1.3]
    assert all(
        [x for x, _ in by_series[name]] == [0.7, 1.0, 1.3]
        for name in ["3B/DART", "3B/DART-pool", "8B/ARC-ch"]
    )
    return by_channel, summary_lookup, by_series


def jitter_offsets(n: int, span: float = 0.20) -> list[float]:
    if n == 1:
        return [0.0]
    return [-span + 2 * span * index / (n - 1) for index in range(n)]


def format_signed(value: float) -> str:
    text = f"{value:.1f}"
    return text.replace("-", "−")


def draw_recovery(
    ax: plt.Axes,
    by_channel: dict[str, list[dict[str, str]]],
    medians: dict[str, float],
) -> None:
    y_lookup = {channel: len(CHANNEL_ORDER) - 1 - index for index, channel in enumerate(CHANNEL_ORDER)}

    ax.axvspan(
        -4,
        4,
        color=PALETTE["band_aqua"],
        alpha=0.75,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(
        0,
        color=PALETTE["control_dark"],
        linewidth=0.95,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )
    for channel in CHANNEL_ORDER:
        y = y_lookup[channel]
        rows = by_channel[channel]
        clean = sorted(
            (
                float(row["recovery_percent"])
                for row in rows
                if row["include_clean_median"] == "1"
            )
        )
        flagged = [
            float(row["recovery_percent"])
            for row in rows
            if row["include_clean_median"] == "0"
        ]
        ax.scatter(
            clean,
            [y + offset for offset in jitter_offsets(len(clean))],
            s=20,
            color=PALETTE["reachability_soft"],
            edgecolor=PALETTE["reachability_mid"],
            linewidth=0.5,
            zorder=2,
        )
        if flagged:
            ax.scatter(
                flagged,
                [y - 0.27] * len(flagged),
                s=36,
                facecolor=PALETTE["white"],
                edgecolor=PALETTE["realized"],
                linewidth=1.25,
                zorder=4,
            )
        median = medians[channel]
        ax.scatter(
            [median],
            [y],
            s=44,
            marker="D",
            color=PALETTE["reachability"],
            edgecolor=PALETTE["white"],
            linewidth=0.55,
            zorder=5,
        )
        ax.text(
            median + 3.1,
            y + 0.27,
            f"median {format_signed(median)}",
            fontsize=8.4,
            color=PALETTE["reachability"],
            fontweight="bold",
            ha="left",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": PALETTE["white"],
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    ax.set_xlim(-82, 82)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xticks([-80, -40, 0, 40, 80])
    ax.set_yticks(
        [y_lookup[channel] for channel in CHANNEL_ORDER],
        [CHANNEL_LABELS[channel] for channel in CHANNEL_ORDER],
    )
    ax.set_xlabel("Oracle–deployment gap recovered (%)", fontsize=9.0)
    ax.tick_params(axis="both", length=2.8, width=0.8, pad=2)
    ax.set_title(
        "Answer-blind selectors recover little of the gap",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    add_panel_label(ax, "a", x=-0.13, y=1.045, fontsize=10.8)
    ax.text(
        0.01,
        0.99,
        "worse than greedy",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="top",
        fontstyle="italic",
    )
    ax.text(
        0.99,
        0.99,
        "gap recovered",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="right",
        va="top",
        fontstyle="italic",
    )
    ax.text(
        0.99,
        0.05,
        "common $K=32$",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="right",
        va="bottom",
    )


def draw_temperature(
    ax: plt.Axes,
    by_series: dict[str, list[tuple[float, float]]],
) -> None:
    for name, style in SERIES_STYLE.items():
        points = by_series[name]
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            markersize=4.2,
            markeredgecolor=PALETTE["white"],
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.annotate(
            style["label"],
            xy=(x[-1], y[-1]),
            xytext=(5, style["label_offset"]),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=style["color"],
            ha="left",
            va="center",
            clip_on=False,
        )

    ax.set_xlim(0.65, 1.67)
    ax.set_ylim(0, 50)
    ax.set_xticks([0.7, 1.0, 1.3])
    ax.set_yticks([0, 20, 40])
    ax.set_xlabel("Sampling temperature", fontsize=9.0)
    ax.set_ylabel("Oracle − majority gap (%)", fontsize=9.0)
    ax.tick_params(axis="both", length=2.8, width=0.8, pad=2)
    ax.set_title(
        "Higher temperature widens the gap",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    add_panel_label(ax, "b", x=-0.18, y=1.045, fontsize=10.8)
    ax.text(
        0.02,
        0.04,
        "8B/HellaSwag: $t=0.7$ not measured",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="bottom",
    )


def recovery_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["reachability_soft"],
            markeredgecolor=PALETTE["reachability_mid"],
            markersize=4.8,
            label="Clean cell",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="D",
            markerfacecolor=PALETTE["reachability"],
            markeredgecolor=PALETTE["white"],
            markersize=5.0,
            label="Clean median",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["realized"],
            markeredgewidth=1.2,
            markersize=5.2,
            label="Format-collapse diagnostic",
        ),
    ]


def main() -> None:
    by_channel, medians, by_series = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.25, 1.15],
        left=0.115,
        right=0.97,
        bottom=0.22,
        top=0.79,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    draw_recovery(ax_a, by_channel, medians)
    draw_temperature(ax_b, by_series)

    fig.legend(
        handles=recovery_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.35, 0.98),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        fontsize=8.0,
    )

    outputs = save_publication_figure(fig, OUTPUT, dpi=600)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
