#!/usr/bin/env python3
"""Build the eight machine-readable source tables used in the main paper."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "table-source"
CONTRACTS = ROOT / "data" / "processed" / "contracts"


def load(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert rows, f"no rows for {path.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    assert abs(actual - expected) <= tolerance, (actual, expected)


def pct(value: float) -> float:
    # Saved summaries may truncate exact count ratios at four decimals.
    # The epsilon preserves conventional half-up paper rounding at x.x5.
    return round(100 * value + 1e-8, 1)


def validate_direct_sources() -> None:
    # Ordinary sampling: these are the final n=150, K=8 files. The similarly
    # named *_K8 files for three models are earlier n=100 runs.
    sampling = {
        "data/raw/rlvr/reroll_reroll_l318b_gsm8k.json": (0.133, 0.187, 0.447),
        "data/raw/rlvr/reroll_reroll_l3b_gsm8k_t768.json": (0.127, 0.153, 0.400),
        "data/raw/rlvr/reroll_reroll_q257_gsm8k.json": (0.227, 0.207, 0.320),
        "data/raw/rlvr/reroll_reroll_q2514_gsm8k.json": (0.327, 0.327, 0.353),
        "data/raw/rlvr/reroll_reroll_l318b_arc.json": (0.840, 0.860, 0.927),
        "data/raw/rlvr/reroll_reroll_q257_arc.json": (0.907, 0.933, 0.960),
    }
    for path, expected in sampling.items():
        data = load(path)
        assert data["config"]["n"] == 150
        assert data["config"]["K"] == 8
        close(data["config"]["temp"], 0.8)
        actual = (
            data["acc"]["single"],
            data["acc"]["majority@K"],
            data["acc"]["oracle@K"],
        )
        for observed, target in zip(actual, expected):
            close(observed, target)

    locus = {
        "data/raw/mlp/battery_hin_g431.json": (144, 120, 90.0, 0.0, 80.0, 120, 4.2),
        "data/raw/mlp/battery_hin_g412.json": (106, 106, 85.9, 23.6, 62.3, 150, 12.0),
        "data/raw/mlp/battery_hin_g4qat.json": (118, 118, 83.1, 4.2, 67.0, 150, 10.7),
        "data/raw/mlp/battery_q2505.json": (103, 40, 87.5, 17.5, 75.0, 34, 2.9),
        "data/raw/mlp/battery_hin_l3b.json": (145, 145, 67.6, 33.8, 78.6, 150, 1.3),
        "data/raw/mlp/battery_hin_prl318.json": (218, 150, 92.0, 36.0, 90.7, 150, 0.0),
    }
    for path, expected in locus.items():
        data = load(path)
        healthy = data["panelC_selectivity"]["buckets"]["R1G1"]
        actual = (
            data["config"]["n_R1G0_sick"],
            data["config"]["n_sick_used"],
            pct(data["panelA_necessity"]["zero_mlp"]["correct"]),
            pct(data["panelA_necessity"]["zero_attn"]["correct"]),
            pct(data["panelA_necessity"]["zero_layer"]["correct"]),
            healthy["n"],
            pct(healthy["zero_mlp"]["right_to_wrong_rate"]),
        )
        assert actual == expected, (path, actual, expected)

    patches = {
        "data/raw/mlp/patch_g412_L29_logiqa.json": ("-4.0v", 95.0, "+4.0v", 97.5),
        "data/raw/mlp/patch_patch_g4qat_L29_logiqa.json": ("-8.0v", 97.5, "+4.0v", 95.0),
        "data/raw/mlp/patch_patch_g431_L39_logiqa.json": ("-4.0v", 92.5, "+2.0v", 27.5),
        "data/raw/mlp/patch_l3b_L20_arc.json": ("-8.0v", 91.7, "+8.0v", 32.5),
        "data/raw/mlp/patch_patch_q2515_L18_piqa.json": ("-4.0v", 42.5, "+8.0v", 90.0),
        "data/raw/mlp/patch_patch_q3b_L28_piqa.json": ("-8.0v", 70.0, "+8.0v", 10.0),
    }
    for path, (rescue_dose, rescue, failure_dose, failure) in patches.items():
        data = load(path)
        close(pct(data["rescue_sick"][rescue_dose]["correct"]), rescue)
        close(pct(1 - data["induce_healthy"][failure_dose]["correct"]), failure)

    fate = {
        "data/raw/rlvr/reroll_fateHi_q32base_dart_t08.json": (300, 200, 0.550, 0.960),
        "data/raw/rlvr/reroll_fateHi_dapo32_dart_t08.json": (300, 200, 0.697, 0.827),
        "data/raw/rlvr/reroll_fateHi_srbase_dart_t08.json": (300, 200, 0.683, 0.940),
        "data/raw/rlvr/reroll_fateHi_srzero_dart_t08.json": (300, 200, 0.803, 0.933),
        "data/raw/rlvr/reroll_q3535b_dart_t08.json": (150, 64, 0.720, 0.833),
        "data/raw/rlvr/reroll_q3535i_dart_t08.json": (150, 64, 0.687, 0.827),
    }
    for path, (n, k, single, oracle) in fate.items():
        data = load(path)
        assert (data["config"]["n"], data["config"]["K"]) == (n, k)
        close(data["acc"]["single"], single)
        close(data["acc"]["oracle@K"], oracle)

    olmo = {
        "data/raw/rlvr/reroll_rlvr_olmo3base_dart_t08.json": (0.127, 0.027, 0.740),
        "data/raw/rlvr/reroll_rlvr_olmo3rlzm_dart_t08.json": (0.253, 0.300, 0.607),
        "data/raw/rlvr/reroll_rlvr_olmo3tsft_dart_t08.json": (0.620, 0.653, 0.820),
        "data/raw/rlvr/reroll_rlvr_olmo3inst_dart_t08.json": (0.673, 0.667, 0.853),
    }
    for path, expected in olmo.items():
        data = load(path)
        actual = (
            data["acc"]["single"],
            data["acc"]["majority@K"],
            data["acc"]["oracle@K"],
        )
        for observed, target in zip(actual, expected):
            close(observed, target)


TABLES: dict[str, list[dict[str, Any]]] = {
    "tab-reroll.csv": [
        {
            "channel": channel,
            "task": task,
            "change_pp": change,
            "budget": budget,
            "selection": selection,
            "verification": verification,
        }
        for channel, budget, selection, verification, values in [
            (
                "learned router",
                "single routed pass",
                "answer-blind",
                "recomputed from eval_gen baseline and routed summaries",
                {"ARC-C": 0.1, "ARC-E": -0.1, "DART": -0.1, "DART-pool": 0.5},
            ),
            (
                "random vote",
                "K=16",
                "answer-blind",
                "final reported values; endpoint outputs included",
                {"ARC-C": -2.0, "ARC-E": 0.0, "DART": -8.8, "DART-pool": -10.0},
            ),
            (
                "MCTS oracle",
                "50 simulations",
                "answer-aware",
                "recomputed from w2_oracle summaries",
                {"ARC-C": 14.3, "ARC-E": 8.7, "DART": 6.2, "DART-pool": 4.0},
            ),
            (
                "random oracle",
                "200 candidates",
                "answer-aware",
                "recomputed from w4_null/fill summaries",
                {"ARC-C": 33.7, "ARC-E": 15.3, "DART": 38.0, "DART-pool": 50.0},
            ),
        ]
        for task, change in values.items()
    ],
    "tab-sampreroll.csv": [
        {
            "task": task,
            "model": model,
            "n": 150,
            "K": 8,
            "temperature": 0.8,
            "deployed_pct": deployed,
            "vote_at_8_pct": vote,
            "oracle_at_8_pct": oracle,
            "gap_pp": gap,
            "source_file": source,
        }
        for task, model, deployed, vote, oracle, gap, source in [
            ("GSM8K", "Llama-3.1-8B", 13.3, 18.7, 44.7, 26.0, "reroll_reroll_l318b_gsm8k.json"),
            ("GSM8K", "Llama-3.2-3B", 12.7, 15.3, 40.0, 24.7, "reroll_reroll_l3b_gsm8k_t768.json"),
            ("GSM8K", "Qwen2.5-7B", 22.7, 20.7, 32.0, 11.3, "reroll_reroll_q257_gsm8k.json"),
            ("GSM8K", "Qwen2.5-14B", 32.7, 32.7, 35.3, 2.6, "reroll_reroll_q2514_gsm8k.json"),
            ("ARC-C", "Llama-3.1-8B", 84.0, 86.0, 92.7, 6.7, "reroll_reroll_l318b_arc.json"),
            ("ARC-C", "Qwen2.5-7B", 90.7, 93.3, 96.0, 2.7, "reroll_reroll_q257_arc.json"),
        ]
    ],
    "tab-locus.csv": [
        {
            "case": case,
            "task": task,
            "layer": layer,
            "sick_eligible": eligible,
            "sick_evaluated": evaluated,
            "mlp_repair_pct": mlp,
            "attention_repair_pct": attention,
            "whole_layer_repair_pct": whole,
            "healthy_n": healthy_n,
            "healthy_break_pct": healthy_break,
            "source_file": source,
        }
        for case, task, layer, eligible, evaluated, mlp, attention, whole, healthy_n, healthy_break, source in [
            ("G4-31B", "AGIEval-LogiQA", 39, 144, 120, 90.0, 0.0, 80.0, 120, 4.2, "battery_hin_g431.json"),
            ("G4-12B", "AGIEval-LogiQA", 29, 106, 106, 85.9, 23.6, 62.3, 150, 12.0, "battery_hin_g412.json"),
            ("G4-12B-QAT", "AGIEval-LogiQA", 29, 118, 118, 83.1, 4.2, 67.0, 150, 10.7, "battery_hin_g4qat.json"),
            ("Qwen2.5-0.5B", "PIQA", 10, 103, 40, 87.5, 17.5, 75.0, 34, 2.9, "battery_q2505.json"),
            ("Llama-3.2-3B", "ARC-C", 20, 145, 145, 67.6, 33.8, 78.6, 150, 1.3, "battery_hin_l3b.json"),
            ("Llama-3.1-8B-pruned", "PIQA", 20, 218, 150, 92.0, 36.0, 90.7, 150, 0.0, "battery_hin_prl318.json"),
        ]
    ],
    "tab-bidir.csv": [
        {
            "case": case,
            "task": task,
            "layer": layer,
            "sick_evaluated": sick,
            "healthy_evaluated": healthy,
            "rescue_pct": rescue,
            "rescue_dose": rescue_dose,
            "failure_pct": failure,
            "failure_dose": failure_dose,
            "source_file": source,
        }
        for case, task, layer, sick, healthy, rescue, rescue_dose, failure, failure_dose, source in [
            ("G4-12B", "AGIEval-LogiQA", 29, 40, 40, 95.0, "-4v", 97.5, "+4v", "patch_g412_L29_logiqa.json"),
            ("G4-12B-QAT", "AGIEval-LogiQA", 29, 40, 40, 97.5, "-8v", 95.0, "+4v", "patch_patch_g4qat_L29_logiqa.json"),
            ("G4-31B", "AGIEval-LogiQA", 39, 40, 40, 92.5, "-4v", 27.5, "+2v", "patch_patch_g431_L39_logiqa.json"),
            ("Llama-3.2-3B", "ARC-C", 20, 24, 40, 91.7, "-8v", 32.5, "+8v", "patch_l3b_L20_arc.json"),
            ("Qwen2.5-1.5B", "PIQA", 18, 40, 40, 42.5, "-4v", 90.0, "+8v", "patch_patch_q2515_L18_piqa.json"),
            ("Qwen2.5-3B", "PIQA", 28, 20, 40, 70.0, "-8v", 10.0, "+8v", "patch_patch_q3b_L28_piqa.json"),
        ]
    ],
    "tab-fate.csv": [
        {
            "pair": pair,
            "training": training,
            "n": n,
            "K": k,
            "base_deployed_pct": base_d,
            "base_oracle_pct": base_o,
            "trained_deployed_pct": trained_d,
            "trained_oracle_pct": trained_o,
            "delta_reachable_count": delta,
            "source_files": source,
        }
        for pair, training, n, k, base_d, base_o, trained_d, trained_o, delta, source in [
            ("Qwen2.5-32B to DAPO-32B", "from-base RLVR", 300, 200, 55.0, 96.0, 69.7, 82.7, -40, "reroll_fateHi_q32base_dart_t08.json|reroll_fateHi_dapo32_dart_t08.json"),
            ("Qwen2.5-Math-7B to SimpleRL-Zero", "from-base RLVR", 300, 200, 68.3, 94.0, 80.3, 93.3, -2, "reroll_fateHi_srbase_dart_t08.json|reroll_fateHi_srzero_dart_t08.json"),
            ("Qwen3.5-35B-A3B Base to Instruct", "full post-training control", 150, 64, 72.0, 83.3, 68.7, 82.7, -1, "reroll_q3535b_dart_t08.json|reroll_q3535i_dart_t08.json"),
        ]
    ],
    "tab-krobust.csv": [
        {
            "pair": pair,
            "K": k,
            "n_matched": 150,
            "base_oracle_pct": base,
            "trained_oracle_pct": trained,
            "gap_pp": gap,
            "delta_reachable_count": delta,
            "verification": "matched-item regrade from included K=200 and K=400 records",
        }
        for pair, k, base, trained, gap, delta in [
            ("DAPO", 200, 95.3, 85.3, 10.0, -15),
            ("DAPO", 400, 96.0, 85.3, 10.7, -16),
            ("SimpleRL", 200, 94.0, 94.7, -0.7, 1),
            ("SimpleRL", 400, 94.0, 94.7, -0.7, 1),
        ]
    ],
    "tab-olmorecipe.csv": [
        {
            "variant": variant,
            "training": training,
            "n": 150,
            "K": k,
            "temperature": 0.8,
            "deployed_pct": deployed,
            "majority_pct": majority,
            "oracle_pct": oracle,
            "source_file": source,
        }
        for variant, training, k, deployed, majority, oracle, source in [
            ("base", "none", 64, 12.7, 2.7, 74.0, "reroll_rlvr_olmo3base_dart_t08.json"),
            ("RL-zero-math", "RLVR", 64, 25.3, 30.0, 60.7, "reroll_rlvr_olmo3rlzm_dart_t08.json"),
            ("thinking-SFT", "SFT", 32, 62.0, 65.3, 82.0, "reroll_rlvr_olmo3tsft_dart_t08.json"),
            ("full-instruct", "SFT+RLHF", 64, 67.3, 66.7, 85.3, "reroll_rlvr_olmo3inst_dart_t08.json"),
        ]
    ],
    "tab-promptcontrol.csv": [
        {
            "task": task,
            "eligible_n": eligible,
            "native_realized_n": native_n,
            "native_pct": native_pct,
            "shared_realized_n": shared_n,
            "shared_pct": shared_pct,
            "retained_pct": retained,
            "native_protocol": "checkpoint-native prompt format",
            "shared_protocol": "same weights with base-prompt format",
        }
        for task, eligible, native_n, native_pct, shared_n, shared_pct, retained in [
            ("GSM8K", 125, 58, 46.4, 24, 19.2, 41.4),
            ("DART", 85, 24, 28.2, 12, 14.1, 50.0),
        ]
    ],
}


TABLE_META = [
    ("tab:reroll", "tab-reroll.csv", "Routing channels and answer-aware selection"),
    ("tab:sampreroll", "tab-sampreroll.csv", "Ordinary-sampling selection gap"),
    ("tab:locus", "tab-locus.csv", "MLP localization and healthy controls"),
    ("tab:bidir", "tab-bidir.csv", "Reciprocal directional interventions"),
    ("tab:fate", "tab-fate.csv", "Matched checkpoint fate comparison"),
    ("tab:krobust", "tab-krobust.csv", "Sampling-budget control"),
    ("tab:olmorecipe", "tab-olmorecipe.csv", "OLMo training-recipe control"),
    ("tab:promptcontrol", "tab-promptcontrol.csv", "Prompt-format control"),
]


PROVENANCE = [
    ("tab:reroll", "data/raw/routing/eval_gen/", "learned-router endpoints"),
    ("tab:reroll", "data/raw/routing/w2_oracle/", "MCTS oracle summaries"),
    ("tab:reroll", "data/raw/routing/w4_null/", "random-path oracle and vote endpoints"),
    ("tab:sampreroll", "data/raw/rlvr/reroll_reroll_*.json", "final n=150 ordinary-sampling summaries"),
    ("tab:locus", "data/raw/mlp/battery_hin_*.json", "high-n intervention panels"),
    ("tab:locus", "data/raw/mlp/battery_q2505.json", "Qwen2.5-0.5B intervention panel"),
    ("tab:bidir", "data/raw/mlp/patch*.json", "direction and dose intervention outputs"),
    ("tab:fate", "data/raw/rlvr/reroll_fateHi_*.json", "matched K=200 checkpoint outputs"),
    ("tab:fate", "data/raw/rlvr/reroll_q3535*.json", "full post-training control"),
    ("tab:krobust", "data/raw/rlvr/reroll_fateHi_*.json", "K=200 per-question records"),
    ("tab:krobust", "data/raw/rlvr/reroll_k400_*.json", "K=400 per-question records"),
    ("tab:olmorecipe", "data/raw/rlvr/reroll_rlvr_olmo3*.json", "lineage summary outputs"),
    ("tab:promptcontrol", "data/raw/rlvr/reroll_rlvr_olmo3*.json", "native-format endpoints"),
    ("tab:promptcontrol", "data/raw/rlvr/reroll_alpharlzm10_*.json", "same-weight shared-format records"),
]


def main() -> None:
    validate_direct_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    CONTRACTS.mkdir(parents=True, exist_ok=True)

    for filename, rows in TABLES.items():
        write_csv(OUT / filename, rows)

    index_rows = [
        {
            "table_label": label,
            "filename": filename,
            "rows": len(TABLES[filename]),
            "description": description,
        }
        for label, filename, description in TABLE_META
    ]
    write_csv(CONTRACTS / "table-index.csv", index_rows)
    write_csv(
        CONTRACTS / "table-provenance.csv",
        [
            {"table_label": label, "source": source, "role": role}
            for label, source, role in PROVENANCE
        ],
    )

    manifest_rows = []
    for _, filename, _ in TABLE_META:
        path = OUT / filename
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        manifest_rows.append(
            {
                "filename": filename,
                "rows": len(records),
                "columns": len(records[0]),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(CONTRACTS / "table-source-manifest.csv", manifest_rows)
    print("Built and source-validated 8 main-paper tables.")


if __name__ == "__main__":
    main()
