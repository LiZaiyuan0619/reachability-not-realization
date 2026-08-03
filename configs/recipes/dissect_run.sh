#!/bin/bash
# open the MLP write (mlp_dissect.py). args: <env> <model> <layer> <task> <tag> [mm]
set -ex
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
source environment/conda/etc/profile.d/conda.sh
ENVN=$1; MODEL=$2; L=$3; TASK=$4; TAG=$5; MM=$6
conda activate environment/conda-envs/$ENVN
cd .; mkdir -p results/lens
EXTRA=""; [ "$MM" = "mm" ] && EXTRA="--mm_model"
python mlp_dissect.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
    --layer "$L" --task "$TASK" --limit 250 $EXTRA --out "results/lens/dissect_${TAG}.json"
chmod -R a+rwX results/lens 2>/dev/null || true
echo "===== dissect done ====="
