# Reachability Is Not Realization

Code for **Reachability Is Not Realization: Tracing the Sources of LLM
Benchmark Gains**(https://arxiv.org/pdf/2608.03219).

This repository contains the routing, MLP intervention, RLVR, training,
analysis, and figure-generation code used in the paper. It also includes the
processed CSV files behind the paper figures and tables. Model weights,
benchmark datasets, and large raw experiment outputs are not included.

## Installation

The main experiment environment is recorded in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Use `requirements-legacy-routing.txt` for the legacy layer-routing drop-ins.
Figure generation only requires the packages in `requirements-figures.txt`.

## Main code

- Routing and selection: `src/fair_budget_oracle_v2.py`,
  `src/fair_budget_analyze.py`, and `src/self_verify.py`
- Evaluation and sampling: `src/eval_drllm.py` and `src/reroll_sc.py`
- MLP interventions: `src/mlp_battery.py`, `src/mlp_dissect.py`,
  `src/mlp_dla.py`, and `src/mlp_patch.py`
- Layer-routing model modifications: `src/data_models/`
- Router training: `src/train_router.py` and `src/train_router_multi.py`

Runnable experiment recipes are provided in `configs/recipes/`.

## Reproduce the paper figures

```bash
pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
```

The generated files are written below `src/figures/final-v2/*/out/`.

## License

The original project code is released under the MIT License. Files that retain
upstream license headers remain subject to those terms.
