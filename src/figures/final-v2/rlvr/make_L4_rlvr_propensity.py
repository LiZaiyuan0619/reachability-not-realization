#!/usr/bin/env python3
"""Draw L4: base sampling propensity and realization after RLVR.

Core conclusion:
Among base-reachable but base-not-produced questions, training most often
realizes answers that the base already samples frequently.

Evidence map:
  single panel — three retained pair/task series across base hit-rate bins,
  with 95% Wilson intervals at every measured point and direct n labels only
  for the hollow n < 5 markers.

Caption facts:
  - Population is restricted to base-reachable, base-not-produced questions.
  - Base sampling uses K=200.
  - Bins are defined by base sampling hit rate and grouped around their
    midpoints with small series-specific offsets on a log-scaled numeric axis.
  - Low bins are thin and the middle bins are not strictly monotone.
  - Missing bins are not imputed or connected through.
  - OR1 and the unrelated temperature-band result are excluded.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE = PACKAGE / "source-data" / "L4_propensity.csv"
OUTPUT = HERE / "out" / "L4_rlvr_propensity"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    PALETTE,
    SINGLE_COLUMN_IN,
    apply_style,
    quiet_grid,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.95
SERIES_ORDER = ["SimpleRL / DART", "SimpleRL / GSM8K", "DAPO / DART"]
SERIES_STYLE = {
    "SimpleRL / DART": {
        "color": PALETTE["reachability"],
        "marker": "o",
        "x_factor": 0.82,
    },
    "SimpleRL / GSM8K": {
        "color": PALETTE["reachability_mid"],
        "marker": "s",
        "x_factor": 1.0,
    },
    "DAPO / DART": {
        "color": PALETTE["realized"],
        "marker": "^",
        "x_factor": 1.22,
    },
}
THIN_LABEL_OFFSETS = {
    ("SimpleRL / DART", 1): (5, 7),
    ("SimpleRL / GSM8K", 2): (-2, 7),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> dict[str, list[dict[str, str]]]:
    rows = read_csv(SOURCE)
    assert len(rows) == 14
    by_series: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_series[row["series"]].append(row)
    assert set(by_series) == set(SERIES_ORDER)
    for series, series_rows in by_series.items():
        series_rows.sort(key=lambda row: int(row["bucket_order"]))
        orders = [int(row["bucket_order"]) for row in series_rows]
        if series == "SimpleRL / GSM8K":
            assert orders == [2, 3, 4, 5]
        else:
            assert orders == [1, 2, 3, 4, 5]
        for row in series_rows:
            total = int(row["total"])
            monetized = int(row["monetized"])
            rate = float(row["rate"])
            assert abs(rate - monetized / total) < 6e-5
            assert int(row["thin_bin"]) == int(total < 5)
    assert sum(int(row["thin_bin"]) for row in rows) == 2
    assert all(
        float(by_series[series][-1]["rate"]) >= 0.85 for series in SERIES_ORDER
    )
    return by_series


def bin_midpoint_percent(row: dict[str, str]) -> float:
    lo = 100 * float(row["lo"])
    hi = 100 * min(float(row["hi"]), 1.0)
    return (lo + hi) / 2


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return 100 * (center - half_width), 100 * (center + half_width)


def draw_propensity(
    ax: plt.Axes,
    by_series: dict[str, list[dict[str, str]]],
) -> None:
    quiet_grid(ax, axis="y")
    ax.axvspan(
        35,
        100,
        color=PALETTE["band_aqua"],
        alpha=0.58,
        linewidth=0,
        zorder=0,
    )

    for series in SERIES_ORDER:
        style = SERIES_STYLE[series]
        rows = by_series[series]
        for row in rows:
            order = int(row["bucket_order"])
            x = bin_midpoint_percent(row) * float(style["x_factor"])
            y = 100 * float(row["rate"])
            successes = int(row["monetized"])
            total = int(row["total"])
            thin = total < 5
            ci_low, ci_high = wilson_interval(successes, total)
            ax.errorbar(
                [x],
                [y],
                yerr=[[y - ci_low], [ci_high - y]],
                fmt="none",
                ecolor=style["color"],
                elinewidth=0.75,
                capsize=1.6,
                capthick=0.70,
                alpha=0.45,
                zorder=2,
            )
            ax.scatter(
                [x],
                [y],
                s=36,
                marker=style["marker"],
                facecolor=PALETTE["white"] if thin else style["color"],
                edgecolor=style["color"],
                linewidth=1.05 if thin else 0.55,
                zorder=4,
            )
            if thin:
                dx, dy = THIN_LABEL_OFFSETS[(series, order)]
                ax.annotate(
                    f"$n={total}$",
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.0,
                    color=style["color"],
                    ha="right" if dx < 0 else "left",
                    va="bottom",
                    zorder=5,
                )

    ax.set_xscale("log")
    ax.set_xlim(0.65, 110)
    ax.set_ylim(-4, 104)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Base hit rate, bin midpoint (%)", fontsize=8.8)
    ax.set_ylabel("Realized after training (%)", fontsize=8.8)
    ax.tick_params(axis="both", length=2.7, width=0.8, pad=2)
    ax.set_title(
        "Base propensity predicts realization",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    ax.text(
        0.02,
        0.975,
        "base-reachable but unproduced · $K=200$",
        transform=ax.transAxes,
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="top",
    )


def add_legend(fig: plt.Figure) -> None:
    handles = []
    for series in SERIES_ORDER:
        style = SERIES_STYLE[series]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                linewidth=0,
                markersize=4.5,
                label=series.replace(" / ", "/"),
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.018),
        ncol=3,
        fontsize=8.0,
        handlelength=1.35,
        handletextpad=0.32,
        columnspacing=0.60,
    )


def main() -> None:
    by_series = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(SINGLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    ax = fig.add_axes([0.180, 0.225, 0.790, 0.620])
    draw_propensity(ax, by_series)
    add_legend(fig)

    outputs = save_publication_figure(fig, OUTPUT)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
