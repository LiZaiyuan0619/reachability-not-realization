#!/bin/bash
# ==============================================================================
# HIGH-N MLP battery (07-14): same mlp_battery.py 3 panels, but --limit and
# --sick_items are PARAMETERS so we can scan the full test set and maximise
# n_sick (the R1G0 case count). Freezes §4's mechanism main table at higher n
# (current n_sick 24-60 -> reviewer-proof). Adds tee log + product validation.
#   bash -ex battery_run_hin.sh <env> <model> <layer> <task> <tag> <limit> <sick> [mm] [rseeds]
# Output: results/lens/battery_<tag>.json
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
echo "===== high-n battery start ====="; echo "host: $(hostname)"; date
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:${PATH}
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
source environment/conda/etc/profile.d/conda.sh

ENVN=$1; MODEL=$2; L=$3; TASK=$4; TAG=$5; LIMIT=${6:-1000}; SICK=${7:-150}; MM=${8:--}; RSEEDS=${9:-20}
conda activate environment/conda-envs/$ENVN
REPO=.
cd "$REPO"; mkdir -p results/lens; nvidia-smi || true
EXTRA=""; [ "$MM" = "mm" ] && EXTRA="--mm_model"
OUTF="results/lens/battery_${TAG}.json"

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$REPO/results/lens" 2>/dev/null || true; }
trap cleanup EXIT
okbatt(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['panelA_necessity']['verdict']['repair_zero_mlp'] is not None" "$1" 2>/dev/null; }

if okbatt "$OUTF"; then echo "SKIP (exists) $OUTF"; else
    python mlp_battery.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --layer "$L" --task "$TASK" --limit "$LIMIT" $EXTRA \
        --rand_seeds "$RSEEDS" --sick_items "$SICK" --bucket_items "$SICK" \
        --out "$OUTF" || echo "FAILED_RUN battery_${TAG}"
fi
okbatt "$OUTF" || { echo "PRODUCT VALIDATION FAILED: battery_${TAG}"; exit 1; }
python3 -c "import json; j=json.load(open('$OUTF')); c=j['config']; print('n_sick=', c.get('n_R1G0_sick') or c.get('n_sick_used'))"
echo "===== high-n battery end (exit 0) ====="
