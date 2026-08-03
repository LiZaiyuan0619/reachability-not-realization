#!/usr/bin/env python3
"""Build the frozen plotting tables for the seven DRLLM main-paper figures.

This script performs data selection and integrity checks only.  It does not draw
figures.  The independent figure session should consume the generated CSVs and
must not silently alter their values or filters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data/processed/figure-source"
PLOT_OUT = HERE / "source-data"
CONTRACTS = ROOT / "data/processed/contracts"
UPSTREAM = ROOT / "data/processed/upstream"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while ignoring blank lines and leading comment lines."""
    with path.open(newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def write_csv(name: str, rows: list[dict], fields: list[str]) -> Path:
    for directory in [OUT, PLOT_OUT]:
        path = directory / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return OUT / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: str) -> float:
    return float(value)


def as_int(value: str) -> int:
    return int(value)


def build_r1() -> list[Path]:
    source = UPSTREAM / "routing/budget_real.csv"
    rows = read_csv(source)
    panel_order = {
        "Llama-3.2-3B / ARC-c": ("a", 1),
        "Llama-3.2-3B / DART": ("b", 2),
        "Qwen3-4B / DART": ("c", 3),
        "Qwen2.5-14B / ARC-c": ("d", 4),
    }
    selected = []
    for row in rows:
        if row["cell"] not in panel_order:
            continue
        panel, order = panel_order[row["cell"]]
        selected.append(
            {
                "panel": panel,
                "panel_order": order,
                "cell": row["cell"],
                "family": row["family"],
                "task": row["task"],
                "K": as_int(row["K"]),
                "random_path_oracle": as_float(row["rand_oracle"]),
                "structured_search_oracle": as_float(row["mcts_oracle"]),
                "deployable_vote": as_float(row["rand_majority"]),
                "baseline": as_float(row["orig"]),
            }
        )
    selected.sort(key=lambda row: (row["panel_order"], row["K"]))
    assert len(selected) == 32
    expected_k = {1, 2, 4, 5, 8, 16, 32, 200}
    for cell in panel_order:
        cell_rows = [row for row in selected if row["cell"] == cell]
        assert {row["K"] for row in cell_rows} == expected_k
        k1 = next(row for row in cell_rows if row["K"] == 1)
        assert k1["random_path_oracle"] == k1["structured_search_oracle"]
        assert k1["random_path_oracle"] == k1["deployable_vote"]
        assert k1["random_path_oracle"] == k1["baseline"]
    path = write_csv(
        "R1_routing_budget.csv",
        selected,
        [
            "panel",
            "panel_order",
            "cell",
            "family",
            "task",
            "K",
            "random_path_oracle",
            "structured_search_oracle",
            "deployable_vote",
            "baseline",
        ],
    )
    return [path]


def model_family(model: str) -> str:
    if model.startswith("Llama"):
        return "Llama"
    if model.startswith("Qwen"):
        return "Qwen"
    return "Mistral"


def build_r2() -> list[Path]:
    matrix_source = UPSTREAM / "routing/heatmap_46cells.csv"
    matrix = read_csv(matrix_source)
    assert len(matrix) == 46
    assert Counter(row["flag"] for row in matrix) == Counter({"ok": 43, "format_collapse": 3})
    assert all(as_float(row["rand_minus_mcts"]) > 0 for row in matrix)

    clean_sorted = sorted(
        (row for row in matrix if row["flag"] == "ok"),
        key=lambda row: as_float(row["rand_minus_mcts"]),
    )
    clean_rank = {
        (row["model"], row["task"]): index + 1 for index, row in enumerate(clean_sorted)
    }
    matrix_out = []
    for row in matrix:
        key = (row["model"], row["task"])
        matrix_out.append(
            {
                "cell_id": f'{row["model"]} / {row["task"]}',
                "model": row["model"],
                "family": model_family(row["model"]),
                "task": row["task"],
                "deploy": as_float(row["deploy"]),
                "structured_search_ceiling": as_float(row["mcts"]),
                "random_path_oracle": as_float(row["rand64"]),
                "random_minus_structured_pp": as_float(row["rand_minus_mcts"]),
                "neg_log10_p_selection_gap": as_float(row["neg_log10_p"]),
                "analysis_status": row["flag"],
                "include_clean_headline": int(row["flag"] == "ok"),
                "clean_rank": clean_rank.get(key, ""),
            }
        )
    matrix_path = write_csv(
        "R2_routing_cells.csv",
        matrix_out,
        [
            "cell_id",
            "model",
            "family",
            "task",
            "deploy",
            "structured_search_ceiling",
            "random_path_oracle",
            "random_minus_structured_pp",
            "neg_log10_p_selection_gap",
            "analysis_status",
            "include_clean_headline",
            "clean_rank",
        ],
    )

    spaces_source = UPSTREAM / "routing/action_spaces.csv"
    spaces = read_csv(spaces_source)
    spaces_out = []
    for row in spaces:
        total = as_int(row["cells_total"])
        wins = as_int(row["cells_random_wins"])
        spaces_out.append(
            {
                "action_space": row["action_space"],
                "cells_total": total,
                "cells_random_wins": wins,
                "win_fraction": wins / total,
                "note": row["note"],
            }
        )
    assert [(row["cells_total"], row["cells_random_wins"]) for row in spaces_out] == [
        (46, 46),
        (23, 23),
        (10, 10),
    ]
    spaces_path = write_csv(
        "R2_action_spaces.csv",
        spaces_out,
        ["action_space", "cells_total", "cells_random_wins", "win_fraction", "note"],
    )

    families_source = UPSTREAM / "routing/search_families.csv"
    families = read_csv(families_source)
    family_out = []
    method_columns = [
        ("random path", "rand_O"),
        ("hill climbing", "hc_O"),
        ("genetic search", "ga_O"),
    ]
    for row in families:
        for method, column in method_columns:
            family_out.append(
                {
                    "model": row["model"],
                    "task": row["task"],
                    "seed": as_int(row["seed"]),
                    "method": method,
                    "oracle_at_64": as_float(row[column]),
                    "baseline": as_float(row["orig"]),
                    "random_path_oracle": as_float(row["rand_O"]),
                }
            )
    assert len(families) == 12
    assert len(family_out) == 36
    for row in families:
        assert as_float(row["rand_O"]) >= as_float(row["hc_O"])
        assert as_float(row["rand_O"]) >= as_float(row["ga_O"])
    families_path = write_csv(
        "R2_search_families.csv",
        family_out,
        [
            "model",
            "task",
            "seed",
            "method",
            "oracle_at_64",
            "baseline",
            "random_path_oracle",
        ],
    )
    return [matrix_path, spaces_path, families_path]


def build_r4() -> list[Path]:
    recovery_source = UPSTREAM / "selection/recovery.csv"
    recovery = read_csv(recovery_source)
    recovery_out = []
    for row in recovery:
        recovery_out.append(
            {
                "cell": row["cell"],
                "channel": row["channel"],
                "recovery_percent": as_float(row["recovery"]),
                "greedy": row["orig"],
                "oracle": row["oracle"],
                "selector": row["selector"],
                "analysis_status": row["flag"],
                "include_clean_median": int(row["flag"] == "ok"),
            }
        )
    recovery_path = write_csv(
        "R4_selector_recovery.csv",
        recovery_out,
        [
            "cell",
            "channel",
            "recovery_percent",
            "greedy",
            "oracle",
            "selector",
            "analysis_status",
            "include_clean_median",
        ],
    )

    by_channel: dict[str, list[float]] = defaultdict(list)
    total_by_channel: Counter[str] = Counter()
    flagged_by_channel: Counter[str] = Counter()
    for row in recovery_out:
        total_by_channel[row["channel"]] += 1
        if row["analysis_status"] == "ok":
            by_channel[row["channel"]].append(float(row["recovery_percent"]))
        else:
            flagged_by_channel[row["channel"]] += 1
    channel_order = [
        "majority vote",
        "confidence",
        "MCTS vote",
        "hill climbing",
        "genetic search",
        "self-verification",
    ]
    summary = []
    for order, channel in enumerate(channel_order, 1):
        values = by_channel[channel]
        summary.append(
            {
                "channel_order": order,
                "channel": channel,
                "n_total": total_by_channel[channel],
                "n_clean": len(values),
                "n_flagged": flagged_by_channel[channel],
                "clean_median_recovery_percent": statistics.median(values),
            }
        )
    expected = {
        "majority vote": -2.22,
        "confidence": -8.89,
        "MCTS vote": -23.33,
        "hill climbing": -23.485,
        "genetic search": -1.855,
        "self-verification": 2.2,
    }
    for row in summary:
        assert abs(float(row["clean_median_recovery_percent"]) - expected[row["channel"]]) < 0.02
    summary_path = write_csv(
        "R4_selector_summary.csv",
        summary,
        [
            "channel_order",
            "channel",
            "n_total",
            "n_clean",
            "n_flagged",
            "clean_median_recovery_percent",
        ],
    )

    temperature_source = UPSTREAM / "selection/temperature.csv"
    temperature = read_csv(temperature_source)
    temperature_out = [
        {
            "cell": row["cell"],
            "temperature": as_float(row["temp"]),
            "oracle": as_float(row["oracle"]),
            "majority": as_float(row["majority"]),
            "selection_gap_pp": as_float(row["gap"]),
        }
        for row in temperature
    ]
    assert len(temperature_out) == 11
    temperature_path = write_csv(
        "R4_temperature.csv",
        temperature_out,
        ["cell", "temperature", "oracle", "majority", "selection_gap_pp"],
    )
    return [recovery_path, summary_path, temperature_path]


def build_m1() -> list[Path]:
    census_source = UPSTREAM / "mlp/census.csv"
    census = read_csv(census_source)
    counts = Counter((row["ckpt"], row["task"]) for row in census)
    seen: Counter[tuple[str, str]] = Counter()
    census_out = []
    for index, row in enumerate(census, 1):
        key = (row["ckpt"], row["task"])
        seen[key] += 1
        genuine = row["genuine"] == "True"
        census_out.append(
            {
                "record_id": index,
                "checkpoint_task_key": f"{row['ckpt']} || {row['task']}",
                "duplicate_count": counts[key],
                "duplicate_index": seen[key],
                "unique_representative": int(seen[key] == 1),
                "checkpoint": row["ckpt"],
                "family": row["family"],
                "task": row["task"],
                "n": row["n"],
                "chance": row["chance"],
                "is_compression": int(row["is_compression"] == "True"),
                "produced_score": as_float(row["G"]),
                "recognition_score": as_float(row["R"]),
                "gap": row["gap"],
                "cell_class": row["cell_class"],
                "genuine_record": int(genuine),
            }
        )
    genuine_rows = [row for row in census if row["genuine"] == "True"]
    assert len(census) == 303
    assert len(genuine_rows) == 35
    assert len({(row["ckpt"], row["task"]) for row in genuine_rows}) == 27
    assert len(counts) == 284
    assert len({row["ckpt"] for row in census}) == 59
    assert len({row["family"] for row in census}) == 11
    census_path = write_csv(
        "M1_census.csv",
        census_out,
        [
            "record_id",
            "checkpoint_task_key",
            "duplicate_count",
            "duplicate_index",
            "unique_representative",
            "checkpoint",
            "family",
            "task",
            "n",
            "chance",
            "is_compression",
            "produced_score",
            "recognition_score",
            "gap",
            "cell_class",
            "genuine_record",
        ],
    )

    scans = [
        ("Llama-3.1-8B, 40% magnitude pruning", "Llama", 40, "piqa", 16, "battery_prl318bs40mL16.json"),
        ("Llama-3.1-8B, 40% magnitude pruning", "Llama", 40, "piqa", 20, "battery_prl318bs40mL20.json"),
        ("Llama-3.1-8B, 40% magnitude pruning", "Llama", 40, "piqa", 24, "battery_prl318bs40mL24.json"),
        ("Qwen2.5-7B, 50% magnitude pruning", "Qwen", 50, "arc_challenge", 12, "battery_prq257s50mL12.json"),
        ("Qwen2.5-7B, 50% magnitude pruning", "Qwen", 50, "arc_challenge", 16, "battery_prq257s50mL16.json"),
        ("Qwen2.5-7B, 50% magnitude pruning", "Qwen", 50, "arc_challenge", 20, "battery_prq257s50mL20.json"),
    ]
    scan_out = []
    lens_root = ROOT / "data/raw/mlp"
    for case, family, sparsity, task, layer, filename in scans:
        data = json.loads((lens_root / filename).read_text())
        config = data["config"]
        verdict = data["panelA_necessity"]["verdict"]
        assert as_int(str(config["layer"])) == layer
        scan_out.append(
            {
                "case": case,
                "family": family,
                "pruning_sparsity_percent": sparsity,
                "task": task,
                "layer": layer,
                "n_sick_total": config["n_R1G0_sick"],
                "n_sick_used": config["n_sick_used"],
                "zmlp_repair": verdict["repair_zero_mlp"],
                "zattn_repair": verdict["repair_zero_attn"],
                "zlayer_repair": verdict["repair_zero_layer"],
                "mlp_over_attn": verdict["ratio_mlp_over_attn"],
                "locus_label": verdict["label"],
                "source_json": filename,
            }
        )
    locked = [0.2414, 0.9655, 0.0, 0.65, 0.725, 0.225]
    assert all(abs(row["zmlp_repair"] - value) < 1e-6 for row, value in zip(scan_out, locked))
    scan_path = write_csv(
        "M1_pruning_layer_scan.csv",
        scan_out,
        [
            "case",
            "family",
            "pruning_sparsity_percent",
            "task",
            "layer",
            "n_sick_total",
            "n_sick_used",
            "zmlp_repair",
            "zattn_repair",
            "zlayer_repair",
            "mlp_over_attn",
            "locus_label",
            "source_json",
        ],
    )
    return [census_path, scan_path]


def build_m2() -> list[Path]:
    necessity = read_csv(UPSTREAM / "mlp/necessity.csv")
    selectivity = read_csv(UPSTREAM / "mlp/selectivity.csv")
    minimality = read_csv(UPSTREAM / "mlp/minimality.csv")
    cases = {"prl318", "g431", "g4qat", "g412", "q2505", "l3b"}
    assert {row["case"] for row in necessity} == cases
    assert {row["case"] for row in selectivity} == cases
    assert {row["case"] for row in minimality} == cases
    assert len(necessity) == 30
    assert len(selectivity) == 6
    assert len(minimality) == 48

    necessity_out = [
        {
            "case": row["case"],
            "label": row["label"],
            "n_sick_total": as_int(row["n_sick"]),
            "intervention": row["intervention"],
            "intervention_key": row["key"],
            "repair_rate": as_float(row["repair"]),
            "n_used": as_int(row["n_used"]),
        }
        for row in necessity
    ]
    necessity_path = write_csv(
        "M2_necessity.csv",
        necessity_out,
        [
            "case",
            "label",
            "n_sick_total",
            "intervention",
            "intervention_key",
            "repair_rate",
            "n_used",
        ],
    )

    selectivity_out = [
        {
            "case": row["case"],
            "label": row["label"],
            "n_sick_total": as_int(row["n_sick"]),
            "repair_sick": as_float(row["repair_sick"]),
            "n_sick_used": as_int(row["n_sick_used"]),
            "break_healthy": as_float(row["break_healthy"]),
            "n_healthy": as_int(row["n_healthy"]),
            "preserve_healthy": as_float(row["preserve_healthy"]),
        }
        for row in selectivity
    ]
    assert max(row["break_healthy"] for row in selectivity_out) <= 0.12
    selectivity_path = write_csv(
        "M2_selectivity.csv",
        selectivity_out,
        [
            "case",
            "label",
            "n_sick_total",
            "repair_sick",
            "n_sick_used",
            "break_healthy",
            "n_healthy",
            "preserve_healthy",
        ],
    )

    minimality_out = []
    for row in minimality:
        full = as_float(row["full_zero"])
        topk = as_float(row["topk"])
        minimality_out.append(
            {
                "case": row["case"],
                "label": row["label"],
                "k": as_int(row["k"]),
                "directional_topk_repair": topk,
                "random_k_repair": as_float(row["randk"]),
                "random_k_sd": as_float(row["randk_sd"]),
                "whole_block_repair": full,
                "relative_to_whole": topk / full if full else "",
                "top32_mass_fraction": as_float(row["frac_mass_top32"]),
            }
        )
    k32 = [row for row in minimality_out if row["k"] == 32]
    assert len(k32) == 6
    assert min(row["top32_mass_fraction"] for row in k32) >= 0.135
    assert max(row["top32_mass_fraction"] for row in k32) <= 0.197
    assert sum(float(row["relative_to_whole"]) >= 0.70 for row in k32) == 4
    minimality_path = write_csv(
        "M2_minimality.csv",
        minimality_out,
        [
            "case",
            "label",
            "k",
            "directional_topk_repair",
            "random_k_repair",
            "random_k_sd",
            "whole_block_repair",
            "relative_to_whole",
            "top32_mass_fraction",
        ],
    )
    return [necessity_path, selectivity_path, minimality_path]


def build_l1() -> list[Path]:
    source = UPSTREAM / "rlvr/fate.csv"
    rows = [row for row in read_csv(source) if row["pair"] in {"SimpleRL", "DAPO"}]
    assert {row["pair"] for row in rows} == {"SimpleRL", "DAPO"}
    assert {(row["pair"], row["task"]) for row in rows} == {
        ("SimpleRL", "dart"),
        ("SimpleRL", "gsm8k"),
        ("DAPO", "dart"),
        ("DAPO", "gsm8k"),
    }
    for row in rows:
        assert as_int(row["K_base"]) == 200
        assert as_int(row["K_trained"]) == 200
    flow_out = []
    for row in rows:
        base_state = row["base_state"]
        trained_state = row["trained_state"]
        if base_state == "latent" and trained_state == "solid":
            flow_class = "gained_from_base_reachable"
        elif base_state == "absent" and trained_state == "solid":
            flow_class = "gained_from_base_unreachable"
        elif base_state == "solid" and trained_state != "solid":
            flow_class = "lost_from_deployed"
        else:
            flow_class = "other"
        flow_out.append(
            {
                "pair": row["pair"],
                "training_kind": row["kind"],
                "task": row["task"],
                "n": as_int(row["n"]),
                "K": as_int(row["K_base"]),
                "base_state": base_state,
                "trained_state": trained_state,
                "flow_class": flow_class,
                "count": as_int(row["count"]),
                "share": as_float(row["share"]),
                "base_deployed": as_float(row["base_deployed"]),
                "trained_deployed": as_float(row["trained_deployed"]),
                "base_oracle": as_float(row["base_oracle"]),
                "trained_oracle": as_float(row["trained_oracle"]),
                "delta_reachable": as_int(row["delta_reachable"]),
            }
        )
    flow_path = write_csv(
        "L1_fate_flows.csv",
        flow_out,
        [
            "pair",
            "training_kind",
            "task",
            "n",
            "K",
            "base_state",
            "trained_state",
            "flow_class",
            "count",
            "share",
            "base_deployed",
            "trained_deployed",
            "base_oracle",
            "trained_oracle",
            "delta_reachable",
        ],
    )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in flow_out:
        grouped[(row["pair"], row["task"])].append(row)
    summary = []
    for (pair, task), group in grouped.items():
        first = group[0]
        gained_reachable = sum(
            row["count"] for row in group if row["flow_class"] == "gained_from_base_reachable"
        )
        gained_unreachable = sum(
            row["count"] for row in group if row["flow_class"] == "gained_from_base_unreachable"
        )
        lost = sum(row["count"] for row in group if row["flow_class"] == "lost_from_deployed")
        summary.append(
            {
                "pair": pair,
                "task": task,
                "n": first["n"],
                "K": first["K"],
                "gained_from_base_reachable": gained_reachable,
                "gained_from_base_unreachable": gained_unreachable,
                "lost_from_deployed": lost,
                "net_deployed_count": gained_reachable + gained_unreachable - lost,
                "base_deployed": first["base_deployed"],
                "trained_deployed": first["trained_deployed"],
                "delta_deployed_pp": 100 * (first["trained_deployed"] - first["base_deployed"]),
                "base_oracle": first["base_oracle"],
                "trained_oracle": first["trained_oracle"],
                "delta_oracle_pp": 100 * (first["trained_oracle"] - first["base_oracle"]),
                "delta_reachable": first["delta_reachable"],
            }
        )
    pair_order = {"SimpleRL": 1, "DAPO": 2}
    task_order = {"gsm8k": 1, "dart": 2}
    summary.sort(key=lambda row: (pair_order[row["pair"]], task_order[row["task"]]))
    assert len(summary) == 4
    assert all(row["gained_from_base_unreachable"] == 0 for row in summary)
    lookup = {(row["pair"], row["task"]): row for row in summary}
    assert lookup[("DAPO", "dart")]["delta_reachable"] == -40
    assert lookup[("SimpleRL", "dart")]["delta_reachable"] == -2
    summary_path = write_csv(
        "L1_fate_summary.csv",
        summary,
        [
            "pair",
            "task",
            "n",
            "K",
            "gained_from_base_reachable",
            "gained_from_base_unreachable",
            "lost_from_deployed",
            "net_deployed_count",
            "base_deployed",
            "trained_deployed",
            "delta_deployed_pp",
            "base_oracle",
            "trained_oracle",
            "delta_oracle_pp",
            "delta_reachable",
        ],
    )
    return [flow_path, summary_path]


def build_l4() -> list[Path]:
    source = UPSTREAM / "rlvr/propensity.csv"
    allowed = {
        ("SimpleRL", "dart"): (1, "SimpleRL / DART"),
        ("SimpleRL", "gsm8k"): (2, "SimpleRL / GSM8K"),
        ("DAPO", "dart"): (3, "DAPO / DART"),
    }
    bucket_order = {"0-2%": 1, "2-5%": 2, "5-15%": 3, "15-35%": 4, ">35%": 5}
    out = []
    for row in read_csv(source):
        key = (row["pair"], row["task"])
        if key not in allowed:
            continue
        series_order, series = allowed[key]
        total = as_int(row["total"])
        out.append(
            {
                "series_order": series_order,
                "series": series,
                "pair": row["pair"],
                "task": row["task"],
                "bucket_order": bucket_order[row["bucket"]],
                "bucket": row["bucket"],
                "lo": as_float(row["lo"]),
                "hi": as_float(row["hi"]),
                "monetized": as_int(row["monetized"]),
                "total": total,
                "rate": as_float(row["rate"]),
                "thin_bin": int(total < 5),
            }
        )
    out.sort(key=lambda row: (row["series_order"], row["bucket_order"]))
    assert len(out) == 14
    assert {row["series"] for row in out} == {
        "SimpleRL / DART",
        "SimpleRL / GSM8K",
        "DAPO / DART",
    }
    path = write_csv(
        "L4_propensity.csv",
        out,
        [
            "series_order",
            "series",
            "pair",
            "task",
            "bucket_order",
            "bucket",
            "lo",
            "hi",
            "monetized",
            "total",
            "rate",
            "thin_bin",
        ],
    )
    return [path]


PURPOSE = {
    "R1_routing_budget.csv": ("R1", "measured matched-budget trajectories"),
    "R2_routing_cells.csv": ("R2", "46 evaluated cells with 43-cell clean-headline flag"),
    "R2_action_spaces.csv": ("R2", "action-space robustness counts"),
    "R2_search_families.csv": ("R2", "raw search-family oracle values"),
    "R4_selector_recovery.csv": ("R4", "cell-level answer-blind recovery"),
    "R4_selector_summary.csv": ("R4", "clean channel medians and denominators"),
    "R4_temperature.csv": ("R4", "measured temperature selection gaps"),
    "M1_census.csv": ("M1", "census records with duplicate and genuine flags"),
    "M1_pruning_layer_scan.csv": ("M1", "two pruning-created layer scans"),
    "M2_necessity.csv": ("M2", "component intervention repair rates"),
    "M2_selectivity.csv": ("M2", "sick repair and healthy-panel break rates"),
    "M2_minimality.csv": ("M2", "directional top-k and random-k curves"),
    "L1_fate_flows.csv": ("L1", "clean K=200 per-question state flows"),
    "L1_fate_summary.csv": ("L1", "four clean pair-task summaries"),
    "L4_propensity.csv": ("L4", "three retained propensity series with thin-bin flag"),
}


def write_manifest(paths: list[Path]) -> Path:
    rows = []
    for path in sorted(paths):
        records = read_csv(path)
        fields = list(records[0].keys()) if records else []
        figure, purpose = PURPOSE[path.name]
        rows.append(
            {
                "figure_id": figure,
                "filename": path.name,
                "purpose": purpose,
                "rows": len(records),
                "columns": len(fields),
                "sha256": sha256(path),
            }
        )
    path = CONTRACTS / "source-data-manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["figure_id", "filename", "purpose", "rows", "columns", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    CONTRACTS.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(build_r1())
    paths.extend(build_r2())
    paths.extend(build_r4())
    paths.extend(build_m1())
    paths.extend(build_m2())
    paths.extend(build_l1())
    paths.extend(build_l4())
    assert len(paths) == len(PURPOSE) == 15
    manifest = write_manifest(paths)
    print(f"Built {len(paths)} frozen CSVs in {OUT}")
    print(f"Wrote manifest: {manifest}")
    for path in sorted(paths):
        print(f"  {path.name}: {len(read_csv(path))} rows  sha256={sha256(path)[:12]}")


if __name__ == "__main__":
    main()
