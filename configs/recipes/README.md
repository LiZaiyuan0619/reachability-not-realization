# Experiment Recipes

These worker scripts preserve the actual command-line parameters and protocol
semantics used for the main experiment families. Local mount points, user
names, and model-cache paths have been replaced during package sanitization.

The scripts are reference recipes, not portable cluster submission wrappers.
Before execution, set the model identifier, environment, output directory,
and compute settings for the target system. The core entry points are:

- routing and selection: `w8_fairbudget_v2.sh`, `self_verify_run.sh`
- ordinary sampling and RLVR: `reroll_run.sh`, `reroll_temp_run.sh`,
  `rlvr_panel.sh`, `consolidate_run.sh`
- MLP localization: `battery_run_hin.sh`, `dissect_run.sh`, `patch_run.sh`,
  `prefill_run.sh`, `readout_run.sh`
- census and broad controls: `w12_mcll_tfnew.sh`, `w12_zmlpsweep.sh`,
  `w13_stage2.sh`, `w15_bench.sh`

Shell comments are retained because they specify budgets, failure checks,
prompt modes, and known comparability boundaries.
