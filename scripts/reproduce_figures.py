#!/usr/bin/env python3
"""Recreate all paper figures from the frozen source-data CSV files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "src" / "figures" / "final-v2"
FIGURE_SCRIPTS = [
    "routing/make_R1_routing_budget.py",
    "routing/make_R2_routing_breadth.py",
    "routing/make_R4_routing_selection.py",
    "mlp/make_M1_mlp_occurrence.py",
    "mlp/make_M2_mlp_localization.py",
    "rlvr/make_L1_rlvr_fate.py",
    "rlvr/make_L4_rlvr_propensity.py",
]


def main() -> None:
    for relative in FIGURE_SCRIPTS:
        script = FIGURE_ROOT / relative
        print(f"[figure] {relative}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print(f"Recreated {len(FIGURE_SCRIPTS)} figures.")


if __name__ == "__main__":
    main()
