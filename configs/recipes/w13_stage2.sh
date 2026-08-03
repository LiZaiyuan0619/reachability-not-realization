#!/bin/bash
# W13 STAGE-2: the paper's main-table heavy eval for ONE confirmed scar case.
# Per task in TASKLIST, on the FULL set (no --limit): baseline gen + mcll, then
# the repair pack at the scar layer L (zmlp / zattn) + an alpha dose curve
# (0.25/0.5/0.75/1.25) — the monotonic-toxicity evidence. Multi-task = proves the
# scar is model-wide, not the one task it was discovered on.
#   bash -ex w13_stage2.sh <drllm|tfnew> <outtag> <L> <tasklist,csv> <model> [mm]
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

ENVN=$1; OUTTAG=$2; L=$3; TASKLIST=$4; MODEL=$5; MMFLAG=$6
conda activate environment/conda-envs/$ENVN
REPO=.
OUT="$REPO/results/w13_stage2_$OUTTAG"
mkdir -p "$OUT"; cd "$REPO"
EXTRA="--dtype bf16"
[ "$MMFLAG" = "mm" ] && EXTRA="$EXTRA --mm_model"
nvidia-smi || true

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT

okjson(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['summary']['acc_micro'] is not None" "$1" 2>/dev/null; }
diskok(){ [ "$(df --output=avail workspace | tail -1)" -gt 5242880 ]; }

run(){ # task suffix extra-args...
    local T=$1 SUF=$2; shift 2
    diskok || { echo "FATAL: storage <5G free, aborting"; exit 1; }
    if okjson "$OUT/${T}_baseline${SUF}.json"; then echo "SKIP $T $SUF"; return; fi
    python eval_drllm.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$T" --mode baseline $EXTRA "$@" --out_dir "$OUT" \
        || echo "FAILED $T $SUF (continuing)"
}

for T in ${TASKLIST//,/ }; do
    run "$T" ""                                                    # full-set gen baseline
    run "$T" "_mcll" --mc_loglik                                   # loglik reference (gap)
    run "$T" "_zmlp${L}" --zero_sublayer mlp:$L                    # repair operator (main)
    run "$T" "_zattn${L}" --zero_sublayer attn:$L                  # attn control (MLP-local)
    for A in 0.25 0.5 0.75 1.25; do                                # dose curve
        run "$T" "_smlp${L}_${A}" --scale_sublayer mlp:$L:$A
    done
done
echo "===== job end (exit 0) ====="
