#!/bin/bash
# PNS Analysis for Step-Forking-DAPO checkpoints
# Runs both forward and reverse PNS for 8 checkpoints
# Total: 16 tasks across 5 GPUs
#
# Usage: bash PNS_test/launch_step_forking_pns.sh

set -e
cd /home/fuzhizhang.fzz/PNS_RLVR

export PYTHONIOENCODING=UTF-8
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MERGED_DIR="/home/fuzhizhang.fzz/model/merge_models"
OUTPUT_BASE="/home/fuzhizhang.fzz/model/merge_models/pns_results/training_dynamic"
PYTHON="/workspace/anaconda3/envs/verl_fzz/bin/python"
SCRIPT="PNS_test/compute_pns.py"

COMMON_ARGS="--benchmark math500 --num_problems 30 --num_rollouts 5 --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --ablation_mode counterfactual --test_all_steps"

# Available GPUs (0,1,4,6,7 are free)
GPUS=(0 1 4 6 7)
NUM_GPUS=${#GPUS[@]}

# Build task list: forward + reverse for each checkpoint
# Interleave forward/reverse for load balance
TASKS=()
TASK_ARGS=()

for step in 200 400 600 800 1000 1200 1400 1600; do
    TASKS+=("Step-Forking-DAPO-step${step}_fwd")
    TASK_ARGS+=("Step-Forking-DAPO-step${step}||")
    TASKS+=("Step-Forking-DAPO-step${step}_rev")
    TASK_ARGS+=("Step-Forking-DAPO-step${step}||--reverse")
done

TOTAL=${#TASKS[@]}
echo "=================================================="
echo "  Step-Forking-DAPO PNS Analysis"
echo "  Total tasks: ${TOTAL} (8 fwd + 8 rev)"
echo "  GPUs: ${GPUS[*]}"
echo "=================================================="

run_task() {
    local gpu=$1
    local task_info=$2
    
    local ckpt=$(echo "$task_info" | cut -d'|' -f1)
    local extra_args=$(echo "$task_info" | cut -d'|' -f3)
    
    local model_path="${MERGED_DIR}/${ckpt}"
    local output_dir="${OUTPUT_BASE}/${ckpt}"
    local mode="forward"
    local log_suffix="run"
    
    if [[ "$extra_args" == *"--reverse"* ]]; then
        mode="reverse"
        log_suffix="run_reverse"
    fi
    
    mkdir -p "${output_dir}"
    
    echo "[GPU ${gpu}] Starting ${ckpt} (${mode})..."
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${SCRIPT} \
        --model "${model_path}" \
        --output_dir "${output_dir}" \
        ${COMMON_ARGS} \
        ${extra_args} \
        > "${output_dir}/${log_suffix}.log" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu}] Finished ${ckpt} (${mode}) (exit: ${exit_code})"
    return ${exit_code}
}

# Round-robin distribution
echo ""
echo "Assignment:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    local_gpu=${GPUS[$i]}
    my_tasks=()
    for j in $(seq $i $NUM_GPUS $((TOTAL - 1))); do
        my_tasks+=("${TASKS[$j]}")
    done
    echo "  GPU ${local_gpu}: ${my_tasks[*]}"
done

echo ""
echo "Launching workers..."

for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=${GPUS[$i]}
    
    (
        for j in $(seq $i $NUM_GPUS $((TOTAL - 1))); do
            run_task ${gpu} "${TASK_ARGS[$j]}"
        done
    ) &
done

echo "All workers launched. Waiting for completion..."
wait
echo ""
echo "All ${TOTAL} tasks complete!"
echo "Results in: ${OUTPUT_BASE}/"
