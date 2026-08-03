#!/bin/bash
# ==============================================================================
# RLVR LEDGER G/R PANEL runner (07-11). One job = one Olmo-3 ckpt, all 4 tasks.
# Task-type aware (eval_drllm gates mcll/n_shot to MC tasks, cot to COT_TASKS):
#   MC tasks  {arc_challenge, agieval_logiqa}: 3 protocols
#       (1) gen 0-shot short           -> <task>_baseline.json          (G produced)
#       (2) gen 5-shot --base_prompt --offset 5 -> <task>_baseline_fs5_off5.json (G_len alt-path)
#       (3) --mc_loglik                -> <task>_baseline_mcll.json     (R recognized: letter+text)
#   GEN tasks {gsm8k, dart}: only
#       (1) gen 0-shot short           -> <task>_baseline.json          (G produced; direct answer)
#   THINK arm (arg4=think): add, for COT-supported tasks {dart, arc_challenge},
#       (4) gen 0-shot --protocol cot --cot_max_new 2048 -> <task>_baseline_cot.json
#       = the FAIR G for a long-CoT model (short-budget gen crashes into pure <think>
#         with no answer; "generation crash != harm" -> never let it be the headline G).
# base ckpt (no chat_template) auto-falls-back to base_prompt for (1) [eval_drllm fix].
# Results -> results/w11_gap_<tag>/. PRODUCT VALIDATION tail: exit NON-ZERO if any
# EXPECTED product (task-type + think aware) is missing/corrupt.  All 1 GPU, env tfnew.
#   bash -ex rlvr_panel.sh <env> <tag> <model_path> [think]
# ==============================================================================
LOG_ROOT=outputs/logs
RUN_ID=${RUN_ID:-${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)-$(hostname)}}
export RUN_ID
LOG_DIR=${LOG_ROOT}/${RUN_ID}
LOG_FILE=${LOG_DIR}/rank${NODE_RANK:-0}.log
if [ -z "${_RJOB_TEE_OK}" ]; then
    export _RJOB_TEE_OK=1; mkdir -p "${LOG_DIR}"
    echo "[boilerplate] LOG_FILE = ${LOG_FILE}"
    exec bash "$0" "$@" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
fi
set -ex
echo "===== rlvr panel start ====="; echo "host: $(hostname)"; date
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:${PATH}
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
source environment/conda/etc/profile.d/conda.sh

ENVN=$1; TAG=$2; MODEL=$3; THINK=$4
conda activate environment/conda-envs/$ENVN
REPO=.
cd "$REPO"
nvidia-smi || true
OUT="$REPO/results/w11_gap_$TAG"; mkdir -p "$OUT"

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$REPO/results" 2>/dev/null || true; }
trap cleanup EXIT

okjson(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['summary']['acc_micro'] is not None" "$1" 2>/dev/null; }
diskok(){ [ "$(df --output=avail workspace | tail -1)" -gt 5242880 ]; }
run_guard(){ local OUTF=$1; shift
    if okjson "$OUTF"; then echo "SKIP (exists) $OUTF"; return 0; fi
    diskok || { echo "FATAL: storage <5G free, aborting"; exit 1; }
    python eval_drllm.py "$@" || echo "FAILED_RUN $(basename "$OUTF") (continuing)"
}

MC_TASKS=(arc_challenge agieval_logiqa)
GEN_TASKS=(gsm8k dart)
COT_OK=(dart arc_challenge)   # eval_drllm COT_TASKS ∩ our task set
LIM=300
BASE="--model_path $MODEL --tokenizer_path $MODEL --mode baseline --dtype bf16 --out_dir $OUT"

echo "##### PANEL $TAG <- $MODEL  (think=${THINK:-no})"

# --- MC tasks: 3 protocols ---
for T in "${MC_TASKS[@]}"; do
    run_guard "$OUT/${T}_baseline.json"          $BASE --task "$T" --limit $LIM
    run_guard "$OUT/${T}_baseline_fs5_off5.json" $BASE --task "$T" --limit $LIM --n_shot 5 --base_prompt --offset 5
    run_guard "$OUT/${T}_baseline_mcll.json"     $BASE --task "$T" --limit $LIM --mc_loglik
done
# --- GEN tasks: gen 0-shot only ---
for T in "${GEN_TASKS[@]}"; do
    run_guard "$OUT/${T}_baseline.json"          $BASE --task "$T" --limit $LIM
done
# --- THINK arm: fair long-CoT G on cot-supported tasks ---
if [ "$THINK" = "think" ]; then
    for T in "${COT_OK[@]}"; do
        run_guard "$OUT/${T}_baseline_cot.json"  $BASE --task "$T" --limit $LIM --protocol cot --cot_max_new 2048
    done
fi

# ---- product validation: exit NON-ZERO if any EXPECTED product missing/corrupt ----
EXPECT=()
for T in "${MC_TASKS[@]}"; do EXPECT+=("${T}_baseline.json" "${T}_baseline_fs5_off5.json" "${T}_baseline_mcll.json"); done
for T in "${GEN_TASKS[@]}"; do EXPECT+=("${T}_baseline.json"); done
if [ "$THINK" = "think" ]; then for T in "${COT_OK[@]}"; do EXPECT+=("${T}_baseline_cot.json"); done; fi
FAILED=()
for F in "${EXPECT[@]}"; do okjson "$OUT/$F" || FAILED+=("$TAG/$F"); done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "PRODUCT VALIDATION FAILED (${#FAILED[@]} missing/corrupt):"
    printf '  MISSING %s\n' "${FAILED[@]}"
    exit 1
fi
echo "===== rlvr panel end (exit 0, all ${#EXPECT[@]} products validated) ====="
