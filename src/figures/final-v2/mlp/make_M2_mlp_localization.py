#!/usr/bin/env python3
"""Draw M2: causal localization, selectivity and within-block organization.

Core conclusion:
For the predefined direct-answer failures, one identified MLP block per case
is the selective causal locus, while its within-block effect is not uniformly
sparse.

Evidence map:
  a — hero: zMLP repair versus same-layer zAttention and whole-layer control
      in six cases, with intervention/total sick denominators.
  b — selectivity: sick-question repair versus breakage on a separate healthy
      panel, with sick/healthy denominators.
  c — organization: the prespecified k=32 directional-versus-random summary,
      normalized to each case's whole-block repair, highlights four strong and
      two weak cases.

Caption facts:
  - zMLP repair spans 67.6–92.0%; same-layer attention is smaller.
  - Healthy-panel breakage spans 0–12%.
  - Top-32 directional units contain 13.5–19.7% of output mass.
  - Top-32 captures at least 87.5% of whole-block repair in 4/6 cases, while
    g431 and q2505 remain weak; this is not a universal sparse circuit.
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
SOURCE_NECESSITY = PACKAGE / "source-data" / "M2_necessity.csv"
SOURCE_SELECTIVITY = PACKAGE / "source-data" / "M2_selectivity.csv"
SOURCE_MINIMALITY = PACKAGE / "source-data" / "M2_minimality.csv"
OUTPUT = HERE / "out" / "M2_mlp_localization"

sys.path.insert(0, str(PACKAGE / "style"))
from drllm_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    PALETTE,
    apply_style,
    quiet_grid,
    save_publication_figure,
)


FIGURE_HEIGHT_IN = 2.80
CASE_ORDER = ["prl318", "g431", "g4qat", "g412", "q2505", "l3b"]
CASE_LABEL = {
    "prl318": "Llama-3.1-8B pruned · L20",
    "g431": "Gemma-31B · L39",
    "g4qat": "Gemma-12B-QAT · L29",
    "g412": "Gemma-12B · L29",
    "q2505": "Qwen-0.5B · L10",
    "l3b": "Llama-3B · L20",
}
CASE_SHORT = {
    "prl318": "Llama-p",
    "g431": "G31B",
    "g4qat": "G12B-QAT",
    "g412": "G12B",
    "q2505": "Q0.5B",
    "l3b": "Llama-3B",
}
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_data() -> tuple[
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    necessity_rows = read_csv(SOURCE_NECESSITY)
    assert len(necessity_rows) == 30
    necessity: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in necessity_rows:
        necessity[row["case"]][row["intervention_key"]] = row
    assert set(necessity) == set(CASE_ORDER)
    assert all(
        set(rows)
        == {"zero_mlp", "zero_attn", "zero_layer", "project_v", "top32"}
        for rows in necessity.values()
    )

    selectivity_rows = read_csv(SOURCE_SELECTIVITY)
    assert len(selectivity_rows) == 6
    selectivity = {row["case"]: row for row in selectivity_rows}
    assert set(selectivity) == set(CASE_ORDER)

    minimality_rows = read_csv(SOURCE_MINIMALITY)
    assert len(minimality_rows) == 48
    minimality: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in minimality_rows:
        minimality[row["case"]].append(row)
    assert set(minimality) == set(CASE_ORDER)
    expected_k = [1, 2, 4, 8, 16, 32, 64, 128]
    for case, rows in minimality.items():
        rows.sort(key=lambda row: int(row["k"]))
        assert [int(row["k"]) for row in rows] == expected_k
        whole = float(necessity[case]["zero_mlp"]["repair_rate"])
        assert all(abs(float(row["whole_block_repair"]) - whole) < 1e-9 for row in rows)

    zmlp = [
        float(necessity[case]["zero_mlp"]["repair_rate"]) for case in CASE_ORDER
    ]
    attention = [
        float(necessity[case]["zero_attn"]["repair_rate"]) for case in CASE_ORDER
    ]
    healthy_break = [
        float(selectivity[case]["break_healthy"]) for case in CASE_ORDER
    ]
    assert round(min(zmlp), 4) == 0.6759 and round(max(zmlp), 4) == 0.92
    assert all(attn_value < mlp_value for attn_value, mlp_value in zip(attention, zmlp))
    assert min(healthy_break) == 0 and round(max(healthy_break), 4) == 0.12

    top32 = {
        case: next(row for row in rows if row["k"] == "32")
        for case, rows in minimality.items()
    }
    strong = [
        case for case, row in top32.items() if float(row["relative_to_whole"]) >= 0.875
    ]
    weak = [
        case for case, row in top32.items() if float(row["relative_to_whole"]) < 0.5
    ]
    masses = [float(row["top32_mass_fraction"]) for row in top32.values()]
    assert len(strong) == 4
    assert set(weak) == {"g431", "q2505"}
    assert round(min(masses), 3) == 0.135 and round(max(masses), 3) == 0.197
    return necessity, selectivity, minimality


def case_positions() -> dict[str, int]:
    return {
        case: len(CASE_ORDER) - 1 - index
        for index, case in enumerate(CASE_ORDER)
    }


def draw_necessity(
    ax: plt.Axes,
    necessity: dict[str, dict[str, dict[str, str]]],
) -> None:
    y_lookup = case_positions()
    quiet_grid(ax, axis="x")
    for case in CASE_ORDER:
        y = y_lookup[case]
        mlp = 100 * float(necessity[case]["zero_mlp"]["repair_rate"])
        attn = 100 * float(necessity[case]["zero_attn"]["repair_rate"])
        layer = 100 * float(necessity[case]["zero_layer"]["repair_rate"])
        ax.plot(
            [attn, mlp],
            [y, y],
            color=PALETTE["control_light"],
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [attn],
            [y],
            s=31,
            marker="s",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["control_dark"],
            linewidth=1.05,
            zorder=3,
        )
        ax.scatter(
            [layer],
            [y],
            s=37,
            marker="D",
            facecolor=PALETTE["reachability_soft"],
            edgecolor=PALETTE["reachability"],
            linewidth=0.75,
            zorder=3,
        )
        ax.scatter(
            [mlp],
            [y],
            s=42,
            marker="o",
            facecolor=PALETTE["realized"],
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=4,
        )

    labels = []
    for case in CASE_ORDER:
        row = necessity[case]["zero_mlp"]
        denominator = f"$n={int(row['n_used'])}/{int(row['n_sick_total'])}$"
        if case == "prl318":
            labels.append(f"Llama-3.1-8B pruned\nL20 · {denominator}")
        elif case == "g4qat":
            labels.append(f"Gemma-12B-QAT\nL29 · {denominator}")
        else:
            labels.append(f"{CASE_LABEL[case]}\n{denominator}")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([y_lookup[case] for case in CASE_ORDER], labels)
    ax.set_xlabel("Sick questions repaired (%)", fontsize=8.8)
    ax.tick_params(axis="x", length=2.7, width=0.8, pad=2)
    ax.tick_params(axis="y", length=0, pad=3, labelsize=8.0)
    ax.set_title(
        "Identified MLP block localizes repair",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )


def draw_selectivity(
    ax: plt.Axes,
    selectivity: dict[str, dict[str, str]],
) -> None:
    y_lookup = case_positions()
    quiet_grid(ax, axis="x")
    for case in CASE_ORDER:
        row = selectivity[case]
        y = y_lookup[case]
        repair = 100 * float(row["repair_sick"])
        broken = 100 * float(row["break_healthy"])
        ax.plot(
            [broken, repair],
            [y, y],
            color=PALETTE["control_light"],
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [broken],
            [y],
            s=34,
            marker="s",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["loss"],
            linewidth=1.05,
            zorder=3,
        )
        ax.scatter(
            [repair],
            [y],
            s=42,
            marker="o",
            facecolor=PALETTE["realized"],
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=4,
        )
        ax.text(
            1.04,
            y,
            f"{int(row['n_sick_used'])}/{int(row['n_healthy'])}",
            transform=ax.get_yaxis_transform(),
            fontsize=8.0,
            color=PALETTE["control_dark"],
            ha="left",
            va="center",
            clip_on=False,
        )

    ax.text(
        1.04,
        1.035,
        "$n_s/n_h$",
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["control_dark"],
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([y_lookup[case] for case in CASE_ORDER])
    ax.tick_params(axis="x", length=2.7, width=0.8, pad=2)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_xlabel("Outcome (%)", fontsize=8.8)
    ax.set_title(
        "Repair is selective",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )


def draw_minimality(
    ax: plt.Axes,
    minimality: dict[str, list[dict[str, str]]],
) -> None:
    y_lookup = case_positions()
    quiet_grid(ax, axis="x")
    ax.axvline(
        100,
        color=PALETTE["control_dark"],
        linewidth=0.8,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )
    for case in CASE_ORDER:
        row = next(row for row in minimality[case] if row["k"] == "32")
        y = y_lookup[case]
        whole = float(row["whole_block_repair"])
        directional = 100 * float(row["directional_topk_repair"]) / whole
        random = 100 * float(row["random_k_repair"]) / whole
        random_sd = 100 * float(row["random_k_sd"]) / whole
        ax.plot(
            [random, directional],
            [y, y],
            color=PALETTE["control_light"],
            linewidth=2.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.errorbar(
            [random],
            [y],
            xerr=random_sd,
            color=PALETTE["control_mid"],
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            [random],
            [y],
            s=30,
            marker="s",
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["control_mid"],
            linewidth=0.95,
            zorder=3,
        )
        ax.scatter(
            [directional],
            [y],
            s=43,
            marker="o",
            facecolor=(
                PALETTE["realized"]
                if case in {"g431", "q2505"}
                else PALETTE["reachability"]
            ),
            edgecolor=PALETTE["white"],
            linewidth=0.6,
            zorder=4,
        )
        label_on_left = directional >= 96
        ax.annotate(
            f"{directional:.0f}",
            xy=(directional, y),
            xytext=(-5 if label_on_left else 5, 0),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=(
                PALETTE["realized"]
                if case in {"g431", "q2505"}
                else PALETTE["reachability"]
            ),
            ha="right" if label_on_left else "left",
            va="center",
            zorder=5,
        )

    ax.text(
        88,
        4.64,
        "4/6 ≥ 88%",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["reachability"],
        ha="center",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": PALETTE["white"],
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=7,
    )
    ax.text(
        18,
        y_lookup["g431"],
        "top-32 mass: 13.5–19.7%",
        fontsize=8.0,
        color=PALETTE["control_dark"],
        ha="left",
        va="center",
        zorder=7,
    )

    ax.set_xlim(-3, 115)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([y_lookup[case] for case in CASE_ORDER])
    ax.tick_params(axis="x", length=2.7, width=0.8, pad=2)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_xlabel("Whole-block repair recovered (%)", fontsize=8.8)
    ax.set_title(
        "Top-32 is not uniformly sufficient",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )


def add_panel_labels(fig: plt.Figure) -> None:
    for label, x in [("a", 0.120), ("b", 0.420), ("c", 0.680)]:
        fig.text(
            x,
            0.925,
            label,
            fontsize=10.8,
            fontweight="bold",
            color=PALETTE["ink"],
            ha="left",
            va="center",
        )


def add_legends(fig: plt.Figure) -> None:
    handles_a = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["realized"],
            markeredgecolor=PALETTE["white"],
            markersize=5.2,
            label="zMLP",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["control_dark"],
            markersize=4.8,
            label="zAttention",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="D",
            markerfacecolor=PALETTE["reachability_soft"],
            markeredgecolor=PALETTE["reachability"],
            markersize=4.8,
            label="Whole",
        ),
    ]
    handles_b = [
        handles_a[0],
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["loss"],
            markersize=4.8,
            label="Healthy break",
        ),
    ]
    handles_c = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=PALETTE["reachability"],
            markeredgecolor=PALETTE["white"],
            markersize=5.0,
            label="Directional top-32",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markerfacecolor=PALETTE["white"],
            markeredgecolor=PALETTE["control_mid"],
            markersize=4.7,
            label="Random-32 mean ± s.d.",
        ),
    ]
    fig.legend(
        handles=handles_a,
        loc="lower center",
        bbox_to_anchor=(0.280, 0.035),
        ncol=3,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.legend(
        handles=handles_b,
        labels=["Sick repair", "Healthy break"],
        loc="lower center",
        bbox_to_anchor=(0.565, 0.035),
        ncol=2,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=0.7,
    )
    fig.legend(
        handles=handles_c,
        loc="lower center",
        bbox_to_anchor=(0.845, -0.015),
        ncol=1,
        fontsize=8.0,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def main() -> None:
    necessity, selectivity, minimality = load_data()
    apply_style(base_font_size=8.4)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, FIGURE_HEIGHT_IN))
    ax_a = fig.add_axes([0.160, 0.235, 0.260, 0.575])
    ax_b = fig.add_axes([0.460, 0.235, 0.180, 0.575])
    ax_c = fig.add_axes([0.720, 0.235, 0.260, 0.575])

    draw_necessity(ax_a, necessity)
    draw_selectivity(ax_b, selectivity)
    draw_minimality(ax_c, minimality)
    add_panel_labels(fig)
    add_legends(fig)

    outputs = save_publication_figure(fig, OUTPUT)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
