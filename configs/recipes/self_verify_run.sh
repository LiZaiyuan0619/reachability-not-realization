#!/bin/bash
# W8 SELF-VERIFY selector worker (one 1-GPU rjob per (model, task)). FOURTH deployable
# selector for the selection-unrecoverability claim: majority (flat/negative), confidence
# (collapses), router (~baseline) all failed; here the model JUDGES its own stored v2
# candidates ("Is the proposed answer correct?" -> forced-choice Yes/No logprob, ONE
# short full-path forward per unique candidate answer, no generation) and we grade the
# candidate it picks (all-No -> falls back to the greedy default = candidate #1).
#   (1) CPU-validate the pure-python selection logic (self_verify.py --selftest)
#   (2) load the model ONCE (plain AutoModelForCausalLM, STOCK transformers -- the verify
#       pass is full-path, so NO data_models drop-in / PYTHONPATH shadow), then run every
#       (task, arm) combo serially against the fairv2 seed0 audit files
#       -> results/w8_selfverify/sv_<tag>_<task>_<arm>.json
# self_verify.py rebuilds each task's samples via eval_drllm.load_task (same order as
# the v2 run) and hard-asserts alignment against the records before any GPU work.
#   args: <model_path> <tag> <fairbudget_dir> <task_csv> <arms_csv> [k]
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
echo "===== job start ====="; echo "host: $(hostname)"; date
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:${PATH}
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
source environment/conda/etc/profile.d/conda.sh
conda activate environment/conda-envs/drllm

MODEL=$1; TAG=$2; FBDIR=$3; TASKS=$4; ARMS=$5; K=${6:-64}
[ -n "$ARMS" ] || { echo "FATAL: usage <model_path> <tag> <fairbudget_dir> <task_csv> <arms_csv> [k]"; exit 1; }

REPO=.
OUT=$REPO/results/w8_selfverify
mkdir -p "$OUT"; cd "$REPO"
nvidia-smi || true

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT

# CPU-validate the pure-python selection core in the job env before burning GPU time.
python self_verify.py --selftest

# seed0 = the audited seed whose fairv2 files carry per-candidate answers/correct/conf.
FILES=()
IFS=',' read -ra TLIST <<< "$TASKS"
for T in "${TLIST[@]}"; do
    F=$FBDIR/fairv2_${T}_seed0.json
    [ -f "$F" ] || { echo "FATAL: missing $F"; exit 1; }
    FILES+=("$F")
done

python self_verify.py --fairbudget "${FILES[@]}" --arms "$ARMS" --k "$K" \
    --model_path "$MODEL" --tag "$TAG" --out_dir "$OUT"
echo "===== job end (exit 0): out=$OUT ====="
