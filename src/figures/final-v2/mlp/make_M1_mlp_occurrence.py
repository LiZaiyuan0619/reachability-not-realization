#!/usr/bin/env python3
"""Draw M1: bounded occurrence and pruning-created MLP repair.

Core conclusion:
Recognition-generation failures are bounded in the census, and two
pruning-created cases reproduce MLP-sensitive repair with different degrees
of layer localization.

Evidence map:
  a — hero: 303 evaluation records in recognition-versus-generation space,
      with the 35 genuine records highlighted and duplicate-aware counts.
  b — validation: measured zMLP repair at three layers in each of two
      pruning-created cases, retaining both the sharp Llama peak and the
      diffuse Qwen boundary.

Caption facts:
  - The census contains 303 records and 284 unique checkpoint-task cells.
  - The 35 genuine records correspond to 27 unique checkpoint-task cells.
  - Recognition is an auxiliary multiple-choice readout, not sampling
    reachability.
  - Panel b reports the fraction of sick questions repaired by zMLP at each
    measured layer; denominators are shown in the facet titles.
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
SOURCE_CENSUS = PACKAGE / "source-data" / "M1_census.csv"
SOURCE_SCAN = PACKAGE / "source-data" / "M1_pruning_layer_scan.csv"
OUTPUT = HERE / "out" / "M1_mlp_occurrence"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    add_panel_label,
    apply_style,
    quiet_grid,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.50
LLAMA_CASE = "Llama-3.1-8B, 40% magnitude pruning"
QWEN_CASE = "Qwen2.5-7B, 50% magnitude pruning"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    census = read_csv(SOURCE_CENSUS)
    assert len(census) == 303
    assert len({row["checkpoint_task_key"] for row in census}) == 284
    assert sum(int(row["unique_representative"]) for row in census) == 284
    assert len({row["checkpoint"] for row in census}) == 59
    assert len({row["family"] for row in census}) == 11

    genuine = [row for row in census if row["genuine_record"] == "1"]
    assert len(genuine) == 35
    assert len({row["checkpoint_task_key"] for row in genuine}) == 27
    assert sum(int(row["is_compression"]) for row in census) == 38
    assert sum(int(row["is_compression"]) for row in genuine) == 10

    scan_rows = read_csv(SOURCE_SCAN)
    assert len(scan_rows) == 6
    scans: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scan_rows:
        scans[row["case"]].append(row)
    assert set(scans) == {LLAMA_CASE, QWEN_CASE}
    for rows in scans.values():
        rows.sort(key=lambda row: int(row["layer"]))

    llama = scans[LLAMA_CASE]
    qwen = scans[QWEN_CASE]
    assert [int(row["layer"]) for row in llama] == [16, 20, 24]
    assert [int(row["layer"]) for row in qwen] == [12, 16, 20]
    assert [round(float(row["zmlp_repair"]), 4) for row in llama] == [
        0.2414,
        0.9655,
        0.0,
    ]
    assert [round(float(row["zmlp_repair"]), 4) for row in qwen] == [
        0.65,
        0.725,
        0.225,
    ]
    return census, scans


def plot_record_group(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    marker: str,
    genuine: bool,
) -> None:
    if not rows:
        return
    x = [100.0 * float(row["produced_score"]) for row in rows]
    y = [100.0 * float(row["recognition_score"]) for row in rows]
    if genuine:
        ax.scatter(
            x,
            y,
            s=31 if marker == "s" else 25,
            marker=marker,
            color=PALETTE["realized"],
            edgecolor=PALETTE["white"],
            linewidth=0.55,
            alpha=0.92,
            zorder=4,
        )
    else:
        ax.scatter(
            x,
            y,
            s=20 if marker == "s" else 15,
            marker=marker,
            color=PALETTE["control_light"],
            edgecolor=PALETTE["control_mid"],
            linewidth=0.4,
            alpha=0.64,
            zorder=2,
        )


def draw_census(ax: plt.Axes, census: list[dict[str, str]]) -> None:
    ax.plot(
        [0, 100],
        [0, 100],
        color=PALETTE["control_mid"],
        linewidth=0.9,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )

    for genuine in [False, True]:
        for compressed, marker in [(False, "o"), (True, "s")]:
            rows = [
                row
                for row in census
                if (row["genuine_record"] == "1") == genuine
                and (row["is_compression"] == "1") == compressed
            ]
            plot_record_group(ax, rows, marker=marker, genuine=genuine)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Direct-generation score (%)", fontsize=9.0)
    ax.set_ylabel("Auxiliary recognition score (%)", fontsize=9.0)
    ax.tick_params(axis="both", length=2.8, width=0.8, pad=2)
    ax.set_title(
        "Recognition–generation gaps are bounded",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    add_panel_label(ax, "a", x=-0.16, y=1.045, fontsize=10.8)

    ax.text(
        0.035,
        0.965,
        "303 records · 284 unique cells",
        transform=ax.transAxes,
        fontsize=8.1,
        color=PALETTE["control_dark"],
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": PALETTE["white"],
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=6,
    )
    ax.text(
        0.035,
        0.875,
        "35 genuine records · 27 unique cells",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=PALETTE["realized"],
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": PALETTE["white"],
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=6,
    )
    ax.text(
        72,
        76,
        "$y=x$",
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="bottom",
        rotation=36,
    )

    legend = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["control_light"],
            markeredgecolor=PALETTE["control_mid"],
            markersize=4.6,
            label="Other record",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["realized"],
            markeredgecolor=PALETTE["white"],
            markersize=5.0,
            label="Genuine gap",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["control_dark"],
            markersize=4.8,
            label="Compressed checkpoint",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.025),
        fontsize=8.0,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0,
    )


def draw_scan(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    title: str,
    first_label_position: str,
    last_label_side: str,
    show_y: bool,
) -> None:
    layers = [int(row["layer"]) for row in rows]
    repair = [100.0 * float(row["zmlp_repair"]) for row in rows]

    quiet_grid(ax, axis="y")
    ax.plot(
        layers,
        repair,
        color=PALETTE["realized"],
        linewidth=1.65,
        marker="o",
        markersize=5.4,
        markeredgecolor=PALETTE["white"],
        markeredgewidth=0.55,
        zorder=3,
    )
    peak_index = max(range(len(repair)), key=repair.__getitem__)
    ax.scatter(
        [layers[peak_index]],
        [repair[peak_index]],
        s=50,
        marker="o",
        color=PALETTE["realized"],
        edgecolor=PALETTE["reachability"],
        linewidth=1.0,
        zorder=4,
    )
    for index, (layer, value) in enumerate(zip(layers, repair)):
        if index == 0:
            if first_label_position == "below":
                offset = (0, -7)
                ha = "center"
                va = "top"
            else:
                offset = (6, 0)
                ha = "left"
                va = "center"
        elif index == len(layers) - 1:
            if last_label_side == "left":
                offset = (-9, 7)
                ha = "right"
            else:
                offset = (7, 5)
                ha = "left"
            va = "bottom"
        elif value >= 92:
            offset = (-7, 0)
            ha = "right"
            va = "center"
        else:
            offset = (0, 7)
            ha = "center"
            va = "bottom"
        label = "0" if value == 0 else f"{value + 1e-9:.1f}"
        ax.annotate(
            label,
            xy=(layer, value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["realized"],
            ha=ha,
            va=va,
            zorder=5,
        )

    ax.set_xlim(min(layers) - 2.2, max(layers) + 2.2)
    ax.set_ylim(-5, 106)
    ax.set_xticks(layers)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("MLP block (layer)", fontsize=8.5)
    if show_y:
        ax.set_ylabel("Sick-question zMLP repair (%)", fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="both", length=2.6, width=0.75, pad=1.8)
    ax.set_title(
        title,
        loc="center",
        fontsize=8.0,
        fontweight="bold",
        pad=4,
        linespacing=1.03,
    )


def main() -> None:
    census, scans = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.62, 1.0],
        left=0.085,
        right=0.985,
        bottom=0.205,
        top=0.790,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(1, 2, wspace=0.32)
    ax_b1 = fig.add_subplot(right[0, 0])
    ax_b2 = fig.add_subplot(right[0, 1], sharey=ax_b1)

    draw_census(ax_a, census)
    draw_scan(
        ax_b1,
        scans[LLAMA_CASE],
        title="Llama, sharp peak\n40% · PIQA\n$n=29/29$",
        first_label_position="right",
        last_label_side="right",
        show_y=True,
    )
    draw_scan(
        ax_b2,
        scans[QWEN_CASE],
        title="Qwen, diffuse\n50% · ARC-C\n$n=40/66$",
        first_label_position="below",
        last_label_side="left",
        show_y=False,
    )

    add_panel_label(ax_b1, "b", x=-0.38, y=1.045, fontsize=10.8)

    outputs = save_publication_figure(fig, OUTPUT)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
