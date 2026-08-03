#!/bin/bash
# W12 stage-2b: zero-MLP layer sweep via hooks (drop-in-free lesion scan — works
# on any architecture/transformers version; zmlp≈skip established on 3B/Qwen).
#   bash -ex w12_zmlpsweep.sh <model_path> <drllm|tfnew> <outtag> <task> <num_layers> [extra...]
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

MODEL=$1; ENVN=$2; OUTTAG=$3; TASK=$4; NL=$5; shift 5; EXTRA="$@"
conda activate environment/conda-envs/$ENVN
REPO=.
OUT="$REPO/results/w12_zmlp_$OUTTAG"
mkdir -p "$OUT"; cd "$REPO"
nvidia-smi || true

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT

okjson(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['summary']['acc_micro'] is not None" "$1" 2>/dev/null; }
diskok(){ [ "$(df --output=avail workspace | tail -1)" -gt 5242880 ]; }

if okjson "$OUT/${TASK}_baseline.json"; then echo "SKIP baseline (done)"; else
    python eval_drllm.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$TASK" --mode baseline --limit 300 $EXTRA --out_dir "$OUT"
fi
for L in $(seq 0 $((NL-1))); do
    diskok || { echo "FATAL: storage <5G free, aborting"; exit 1; }
    if okjson "$OUT/${TASK}_baseline_zmlp${L}.json"; then echo "SKIP zmlp $L (done)"; continue; fi
    python eval_drllm.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$TASK" --mode baseline --limit 300 --zero_sublayer mlp:$L $EXTRA \
        --out_dir "$OUT" || echo "FAILED zmlp $L (continuing)"
done
echo "===== job end (exit 0) ====="
