#!/bin/bash
# W8 v2: FAIR-BUDGET reroll audit + SELECTION column -- budget-matched oracle@k AND the
# deployable selectors majority@k / confidence@k for MCTS / random-program / output-sampling
# (fair_budget_oracle_v2.py). Same generators / verifier / decode length / samples as
# w8_fairbudget.sh; candidate #1 = full greedy so pass@1==baseline for all arms. v2 stores
# per-candidate answer strings + correctness + mean-logprob (no early-stop -> full kmax set),
# so fair_budget_analyze.py can compute selection gap = oracle@k - majority@k. 1 GPU.
#   bash -ex w8_fairbudget_v2.sh <tasks> [kmax] [seed] [model] [modeling_FULL_file] [tag] \
#                                [action_space=drllm|polar|skiponly] \
#                                [gens=mcts,rand,samp(,hillclimb)] [temp] [out_suffix]
# <tasks> is a COMMA LIST (e.g. "mmlu,hellaswag,piqa" or a single "arc_challenge") run
# SERIALLY under ONE model load -> the 2-5 min model load is amortized across the whole
# list (short MC tasks no longer each reload the model). Each task writes its own output
# file. A single task name still works (1-element list), so old 1-task calls are unchanged.
# tag distinguishes models in the output dir (l3b default / l318b for the 8B arm), so
# 3B and 8B on the same task/seed do NOT collide. Needs the data_models drop-in.
# action_space=polar pins BOTH the mcts and rand arms to PoLar's own program family
# (contiguous segments <=4, <=1 repeat) -> output filename gets a _polar suffix, so
# polar and drllm runs on the same task/seed/tag never collide. gens lets the polar
# tier drop the (action-space-agnostic) samp arm to save GPU. args 7/8 default to the
# original behavior, so all pre-existing 6-arg calls are byte-for-byte unchanged.
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

TASKS=$1   # comma list (serial under one model load) or a single task name
KMAX=${2:-64}
SEED=${3:-0}
MODEL=${4:-models/Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72}
MODELING=${5:-modeling_llama_FULL.py}
TAG=${6:-l3b}
ACTION_SPACE=${7:-drllm}
GENS=${8:-mcts,rand,samp}
TEMP=${9:-0.7}   # sampling-arm temperature (arg 9; pre-existing 8-arg calls unchanged)
OUT_SUFFIX=${10:-}   # arg10: extra output-filename suffix (e.g. _hc / _ga for a single-arm
                     # rerun over already-landed baselines). Default '' == byte-identical to
                     # every pre-existing 9-arg call (fair_budget_oracle_v2.py --out_suffix).
REPO=.
OUT="$REPO/results/w8_fairbudget_v2/$TAG"
mkdir -p "$OUT"; cd "$REPO"
nvidia-smi || true

# ---- data_models drop-in shadow (adds model.model.layer_indices) ----
SITE=$(python -c "import transformers, os; print(os.path.dirname(os.path.dirname(transformers.__file__)))")
# pick the family's stock backup guard (llama by default; 8B is ALSO llama -> same shadow)
FAM=llama; case "$MODELING" in *qwen2*) FAM=qwen2;; *qwen3*) FAM=qwen3;; *mistral*) FAM=mistral;; *nemotron*) FAM=nemotron;; esac
[ -f "$SITE/transformers/models/$FAM/modeling_$FAM.py.STOCKBAK" ] || { echo "FATAL: no STOCKBAK for $FAM"; exit 1; }
SHADOW=/tmp/tfshadow_$$
mkdir -p "$SHADOW"; cp -r "$SITE/transformers" "$SHADOW/"
cp "$REPO/data_models/$MODELING" "$SHADOW/transformers/models/$FAM/modeling_$FAM.py"
export PYTHONPATH="$SHADOW:$PYTHONPATH"
python -c "import transformers; assert transformers.__file__.startswith('$SHADOW'); print('shadow OK (data_models/$MODELING)')"

cleanup(){ chmod -R a+rwX "$LOG_DIR" "$OUT" 2>/dev/null || true; }
trap cleanup EXIT

# CPU-validate the pure-python selector core in the job env before burning GPU time.
python fair_budget_oracle_v2.py --selftest
python fair_budget_analyze.py   --selftest

# arc/MC: limit 300 (first 300 test). dart/dart_pool: 60/level -> 300.
# MAX_NEW (env, not a positional -- 10 positionals is already fragile): pin the decode
# length for every arm. Unset/0 keeps the historical per-task default (dart 15 / else 10),
# so every pre-existing call is byte-for-byte unchanged. Set it when this path arm has to
# be read against the act-3 reroll arm, which decodes at 768-1024: at 15 tokens the
# reasoning checkpoints are cut off before they emit an answer (orig_acc 0.000-0.010 vs
# 0.550-0.803 on the SAME checkpoints), and pass@k over a truncated orig_acc is an artifact.
MAX_NEW=${MAX_NEW:-0}
python fair_budget_oracle_v2.py --model_path "$MODEL" --tasks "$TASKS" \
    --per_level 60 --limit 300 --kmax "$KMAX" --num_simulations 200 \
    --temperature "$TEMP" --seed "$SEED" --gens "$GENS" --max_new "$MAX_NEW" \
    --action_space "$ACTION_SPACE" --out_dir "$OUT" --out_suffix "$OUT_SUFFIX"
echo "===== job end (exit 0) ====="
