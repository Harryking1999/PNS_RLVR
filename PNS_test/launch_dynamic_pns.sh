#!/bin/bash
# Dynamic PNS Analysis: Run PNS on DAPO and Forking-DAPO checkpoints
# Uses a GPU queue to efficiently parallelize across available GPUs
#
# Usage: bash PNS_test/launch_dynamic_pns.sh

set -e
cd /home/fuzhizhang.fzz/PNS_RLVR

export PYTHONIOENCODING=UTF-8
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MERGED_DIR="/ssdwork/fuzhizhang/merged_models"
OUTPUT_BASE="/ssdwork/fuzhizhang/pns_results/training_dynamic"
PYTHON="/workspace/anaconda3/envs/verl_fzz/bin/python"
SCRIPT="PNS_test/compute_pns.py"

# Common args - use 30 problems and 5 rollouts
COMMON_ARGS="--benchmark math500 --num_problems 30 --num_rollouts 5 --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --ablation_mode counterfactual --test_all_steps"

# Available GPUs
GPUS=(0 1 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# Build checkpoint list: DAPO + Forking-DAPO, interleaved for load balance
# Early checkpoints are fast, late ones are slow
# Interleave: DAPO-early, Forking-early, DAPO-late, Forking-late
CHECKPOINTS=()

# DAPO checkpoints (all 13)
for step in 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2550; do
    CHECKPOINTS+=("DAPO-step${step}")
done

# Forking-DAPO checkpoints (all 13)
for step in 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2550; do
    CHECKPOINTS+=("Forking-DAPO-step${step}")
done

TOTAL=${#CHECKPOINTS[@]}
echo "=================================================="
echo "  Dynamic PNS Analysis"
echo "  Total checkpoints: ${TOTAL}"
echo "  GPUs: ${GPUS[*]}"
echo "=================================================="

# Function to run a single checkpoint on a given GPU
run_checkpoint() {
    local gpu=$1
    local ckpt=$2
    local model_path="${MERGED_DIR}/${ckpt}"
    local output_dir="${OUTPUT_BASE}/${ckpt}"
    
    mkdir -p "${output_dir}"
    
    echo "[GPU ${gpu}] Starting ${ckpt}..."
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${SCRIPT} \
        --model "${model_path}" \
        --output_dir "${output_dir}" \
        ${COMMON_ARGS} \
        > "${output_dir}/run.log" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu}] Finished ${ckpt} (exit: ${exit_code})"
    return ${exit_code}
}

# Queue-based parallel execution
# Each GPU runs checkpoints sequentially, all GPUs run in parallel
split_checkpoints() {
    # Distribute checkpoints to GPUs in round-robin (interleaved for balance)
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local gpu_idx=$i
        local gpu=${GPUS[$gpu_idx]}
        
        # Collect checkpoints for this GPU
        local my_ckpts=()
        for j in $(seq $i $NUM_GPUS $((TOTAL - 1))); do
            my_ckpts+=("${CHECKPOINTS[$j]}")
        done
        
        echo "GPU ${gpu}: ${my_ckpts[*]}"
        
        # Launch background worker for this GPU
        (
            for ckpt in "${my_ckpts[@]}"; do
                run_checkpoint ${gpu} "${ckpt}"
            done
        ) &
    done
}

echo ""
echo "Assignment:"
split_checkpoints

echo ""
echo "All workers launched. Waiting for completion..."
wait
echo ""
echo "🎉 All ${TOTAL} checkpoints complete!"
echo "Results in: ${OUTPUT_BASE}/"
