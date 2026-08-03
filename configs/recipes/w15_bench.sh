#!/bin/bash
# W15 COMMON-BENCHMARK evaluation (the field-standard set Dr.LLM/PoLar use).
# For a confirmed scar case @L, per task: direct-gen baseline (deployment number),
# mcll loglik (the "knows" ceiling = gap), zmlp repair. This tests the user's key
# question: do the anchor cases gain on COMMON benchmarks (MMLU/GSM8K/ARC/HellaSwag/
# TruthfulQA/...), not just the discovery task? And measures the gap the field's way.
#   bash -ex w15_bench.sh <drllm|tfnew> <outtag> <L> <model> [mm]
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

ENVN=$1; OUTTAG=$2; L=$3; MODEL=$4; MMFLAG=$5
conda activate environment/conda-envs/$ENVN
REPO=.
OUT="$REPO/results/w15_bench_$OUTTAG"
mkdir -p "$OUT"; cd "$REPO"
EXTRA="--dtype bf16"; [ "$MMFLAG" = "mm" ] && EXTRA="$EXTRA --mm_model"
LIM="--limit 2000"   # cap huge tasks (hellaswag=10042) for a fast first-pass gain table
nvidia-smi || true
cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT
okjson(){ python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['summary']['acc_micro'] is not None" "$1" 2>/dev/null; }
diskok(){ [ "$(df --output=avail workspace | tail -1)" -gt 5242880 ]; }
run(){ local T=$1 SUF=$2; shift 2
    diskok || { echo "FATAL: storage <5G free"; exit 1; }
    if okjson "$OUT/${T}_baseline${SUF}.json"; then echo "SKIP $T $SUF"; return; fi
    python eval_drllm.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
        --task "$T" --mode baseline $EXTRA $LIM "$@" --out_dir "$OUT" || echo "FAILED $T $SUF (continuing)"
}
# MCQ tasks: gen (deploy) + mcll (knows/gap) + zmlp (repair)
for T in mmlu arc_challenge hellaswag piqa openbookqa commonsense_qa winogrande truthfulqa boolq; do
    run "$T" ""
    run "$T" "_mcll"   --mc_loglik
    run "$T" "_zmlp${L}" --zero_sublayer mlp:$L
done
# Generative numeric (no loglik): gen + zmlp only
for T in gsm8k; do
    run "$T" ""
    run "$T" "_zmlp${L}" --zero_sublayer mlp:$L
done
echo "===== job end (exit 0) ====="
