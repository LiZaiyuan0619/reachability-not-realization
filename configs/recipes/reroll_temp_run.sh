#!/bin/bash
# ==============================================================================
# RLVR LEDGER REROLL runner (07-11). One job = one (ckpt, task) at TWO temperatures
# (0.8 + 1.3) = 2 ledger cells. reroll_sc.py reports per question:
#   single (greedy, deployable) / random@K / majority@K (self-consistency, deployable)
#   / oracle@K (best-of-N with oracle verifier = pass@K, the CHEATING ceiling).
# Ledger use: sampling-reachable = oracle@K ; selection = majority@K - single ;
# temperature sweep answers "is the base>RL sampling ceiling temperature-limited?".
# dart auto-gets subset_per_level=30 (even coverage of MATH difficulty levels).
# base ckpt (no chat_template) uses reroll_sc.chat()'s base_prompt fallback (07-11 patch).
# Outputs -> results/lens/reroll_<tag_base>_t08.json  /  _t13.json .  1 GPU, env tfnew.
#   bash -ex reroll_temp_run.sh <env> <model> <task> <tag_base> <max_new> <limit> <K> [mm]
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
echo "===== rlvr reroll start ====="; echo "host: $(hostname)"; date
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:${PATH}
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
source environment/conda/etc/profile.d/conda.sh

ENVN=$1; MODEL=$2; TASK=$3; TAGB=$4; MAXNEW=${5:-768}; LIMIT=${6:-150}; K=${7:-64}; MM=$8
conda activate environment/conda-envs/$ENVN
REPO=.
cd "$REPO"; mkdir -p results/lens
nvidia-smi || true
EXTRA=""; [ "$MM" = "mm" ] && EXTRA="--mm_model"

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$REPO/results/lens" 2>/dev/null || true; }
trap cleanup EXIT
okreroll(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['acc']['oracle@K'] is not None" "$1" 2>/dev/null; }
diskok(){ [ "$(df --output=avail workspace | tail -1)" -gt 5242880 ]; }

echo "##### REROLL $TAGB task=$TASK K=$K max_new=$MAXNEW limit=$LIMIT <- $MODEL"
for PAIR in "0.8:t08" "1.3:t13"; do
    TEMP="${PAIR%%:*}"; SUF="${PAIR##*:}"
    OUTF="results/lens/reroll_${TAGB}_${SUF}.json"
    if okreroll "$OUTF"; then echo "SKIP (exists) $OUTF"; continue; fi
    diskok || { echo "FATAL: storage <5G free, aborting"; exit 1; }
    python reroll_sc.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$TASK" --K "$K" --temp "$TEMP" --limit "$LIMIT" --max_new "$MAXNEW" \
        $EXTRA --out "$OUTF" || echo "FAILED_RUN reroll_${TAGB}_${SUF} (continuing)"
done

# ---- product validation: both temperature cells must validate ----
FAILED=()
for SUF in t08 t13; do okreroll "results/lens/reroll_${TAGB}_${SUF}.json" || FAILED+=("reroll_${TAGB}_${SUF}"); done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "PRODUCT VALIDATION FAILED (${#FAILED[@]} missing/corrupt):"
    printf '  MISSING %s\n' "${FAILED[@]}"
    exit 1
fi
echo "===== rlvr reroll end (exit 0, both temps validated) ====="
