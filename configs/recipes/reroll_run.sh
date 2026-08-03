#!/bin/bash
# reroll-generalization: self-consistency/best-of-N audit (reroll_sc.py). args: <env> <model> <task> <tag> <max_new> <limit> [mm]
set -ex
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
source environment/conda/etc/profile.d/conda.sh
ENVN=$1; MODEL=$2; TASK=$3; TAG=$4; MAXNEW=${5:-512}; LIMIT=${6:-150}; MM=$7; K=${8:-8}
conda activate environment/conda-envs/$ENVN
cd .; mkdir -p results/lens
EXTRA=""; [ "$MM" = "mm" ] && EXTRA="--mm_model"
python reroll_sc.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
    --task "$TASK" --K "$K" --temp 0.8 --limit "$LIMIT" --max_new "$MAXNEW" $EXTRA --out "results/lens/reroll_${TAG}.json"
chmod -R a+rwX results/lens 2>/dev/null || true
echo "===== reroll_sc done ====="
