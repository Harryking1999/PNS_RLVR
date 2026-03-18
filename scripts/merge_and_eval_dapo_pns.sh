#!/usr/bin/env bash
# Merge and Evaluate DAPO-PNS Checkpoints
# Usage:
#   bash scripts/merge_and_eval_dapo_pns.sh
#   SKIP_MERGE=1 bash scripts/merge_and_eval_dapo_pns.sh
#   SKIP_EVAL=1 bash scripts/merge_and_eval_dapo_pns.sh
#   SKIP_PNS=1 bash scripts/merge_and_eval_dapo_pns.sh
#   EVAL_GPU=1 bash scripts/merge_and_eval_dapo_pns.sh
set -euo pipefail
cd /home/fuzhizhang.fzz/PNS_RLVR

PYTHON="/workspace/anaconda3/envs/verl_fzz/bin/python"
CKPT_ROOT="/ssdwork/fuzhizhang/ckpts/DAPO/RLVR-Qwen3-1.7B-DAPO-PNS"
MERGE_DIR="/ssdwork/fuzhizhang/merged_models"
MODEL_PREFIX="DAPO-PNS"
STEPS=(200 400 600 737 800 850)
EVAL_GPU="${EVAL_GPU:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_PNS="${SKIP_PNS:-0}"
EVAL_OUTPUT="${MERGE_DIR}/eval_results.json"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "======== DAPO-PNS Merge+Eval Pipeline ========"
echo "  STEPS: ${STEPS[*]}  GPU: ${EVAL_GPU}"
mkdir -p "${MERGE_DIR}"

if [ "${SKIP_MERGE}" = "0" ]; then
    echo "===== Phase 1: Merge FSDP ====="
    for STEP in "${STEPS[@]}"; do
        ACTOR_DIR="${CKPT_ROOT}/global_step_${STEP}/actor"
        TARGET_DIR="${MERGE_DIR}/${MODEL_PREFIX}-step${STEP}"
        if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/config.json" ]; then
            echo "  [step ${STEP}] Already merged, skip"
            continue
        fi
        echo "  Merging step ${STEP}..."
        ${PYTHON} -m verl.model_merger merge --backend fsdp \
            --local_dir "${ACTOR_DIR}" --target_dir "${TARGET_DIR}"
        echo "  Step ${STEP} merged!"
    done
fi

if [ "${SKIP_EVAL}" = "0" ]; then
    echo "===== Phase 2: Eval AIME24/AIME25/MATH500 ====="
    MODEL_PATHS=()
    for STEP in "${STEPS[@]}"; do
        MP="${MERGE_DIR}/${MODEL_PREFIX}-step${STEP}"
        if [ -d "${MP}" ]; then MODEL_PATHS+=("${MP}"); fi
    done
    if [ ${#MODEL_PATHS[@]} -gt 0 ]; then
        CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PYTHON} tools/eval_benchmarks.py \
            --models "${MODEL_PATHS[@]}" --benchmarks aime24 aime25 math500 math500_noop \
            --max_tokens 16384 --output "${EVAL_OUTPUT}"
        echo "  Eval done: ${EVAL_OUTPUT}"
    fi
fi

if [ "${SKIP_PNS}" = "0" ]; then
    echo "===== Phase 3: PNS Estimation ====="
    for STEP in 200 600 850; do
        MP="${MERGE_DIR}/${MODEL_PREFIX}-step${STEP}"
        if [ ! -d "${MP}" ]; then continue; fi
        echo "  PNS: step${STEP}"
        CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PYTHON} PNS_test/compute_pns.py \
            --model "${MP}" --benchmark math500 --num_problems 50 --num_rollouts 5 \
            --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --resume
    done
    ${PYTHON} PNS_test/compare_pns_results.py
fi

echo "======== Done! ========"
echo "  Merged: ${MERGE_DIR}/"
echo "  Eval:   ${EVAL_OUTPUT}"
echo "  PNS:    /ssdwork/fuzhizhang/pns_results/"
