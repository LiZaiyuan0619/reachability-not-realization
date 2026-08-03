#!/bin/bash
# CPU-only consolidation runner (07-15). Runs the zero-GPU §5 analysis on a worker
# node (gpu=1 only to schedule fast; no GPU used). Writes report to results/lens/.
LOG_ROOT=outputs/logs
RUN_ID=${RUN_ID:-${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)-$(hostname)}}
export RUN_ID; LOG_DIR=${LOG_ROOT}/${RUN_ID}
if [ -z "${_RJOB_TEE_OK}" ]; then
    export _RJOB_TEE_OK=1; mkdir -p "${LOG_DIR}"
    exec bash "$0" "$@" 2>&1 | tee -a "${LOG_DIR}/rank0.log"; exit "${PIPESTATUS[0]}"
fi
set -ex
source environment/conda/etc/profile.d/conda.sh
conda activate environment/conda-envs/tfnew
cd .
python3 debug/consolidate_0715.py
chmod -R a+rwX results/lens 2>/dev/null || true
echo "===== consolidate end (exit 0) ====="
