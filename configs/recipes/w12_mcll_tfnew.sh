#!/bin/bash
# W11: generic mcll-baseline runner for the gap≈gain scatter (realization-gap
# measurement: letter-loglik baseline per model x task; gains already measured).
#   bash -ex w11_mcll.sh <model_path> <llama|qwen2|qwen3|nemotron|mistral> <outtag> <task1,task2,...> [extra...]
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
conda activate environment/conda-envs/tfnew

MODEL=$1; ARCH=$2; OUTTAG=$3; TASKS=$4; shift 4; EXTRA="$@"
REPO=.
OUT="$REPO/results/w11_$OUTTAG"
mkdir -p "$OUT"; cd "$REPO"
nvidia-smi || true

echo 'no shadow (tfnew stock transformers)'

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT

IFS=',' read -ra TLIST <<< "$TASKS"
for T in "${TLIST[@]}"; do
    python eval_drllm.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$T" --mode baseline --limit 300 --mc_loglik $EXTRA --out_dir "$OUT" \
        || echo "FAILED $T (continuing)"
done
echo "===== job end (exit 0) ====="
