#!/bin/bash
# prompting-artifact control (mlp_prefill.py). args: <env> <model> <layer> <specs> <tag> [mm]
set -ex
export HF_HOME=models/cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
source environment/conda/etc/profile.d/conda.sh
ENVN=$1; MODEL=$2; L=$3; SPECS=$4; TAG=$5; MM=$6
conda activate environment/conda-envs/$ENVN
cd .; mkdir -p results/lens
EXTRA=""; [ "$MM" = "mm" ] && EXTRA="--mm_model"
python mlp_prefill.py --model_path "$MODEL" --tokenizer_path "$MODEL" \
    --layer "$L" --eval_specs "$SPECS" --limit 250 $EXTRA --out "results/lens/prefill_${TAG}.json"
chmod -R a+rwX results/lens 2>/dev/null || true
echo "===== prefill done ====="
