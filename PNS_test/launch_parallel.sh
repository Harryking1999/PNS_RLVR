#!/usr/bin/env bash
#
# 并行 PNS 估算: 4 张 GPU 并行跑 6 个模型
#
# 第一批 (GPU 0-3): 4 个模型并行
# 第二批 (GPU 0-1): 2 个模型并行
#
# 日志在 /ssdwork/fuzhizhang/pns_results/{model_name}/run.log

set -euo pipefail

cd /home/fuzhizhang.fzz/PNS_RLVR

# 使用 verl_fzz conda 环境 (有 tensordict, vllm, torch)
CONDA_PREFIX="/workspace/anaconda3/envs/verl_fzz"
PYTHON="${CONDA_PREFIX}/bin/python"

# vLLM v0.8.5 需要 spawn 模式避免 CUDA fork 错误
export VLLM_WORKER_MULTIPROC_METHOD=spawn

COMMON_ARGS="--benchmark math500 --num_problems 50 --num_rollouts 5 \
--entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --test_all_steps --resume"

run_model() {
    local GPU=$1
    local MODEL=$2
    local MODEL_NAME=$(basename "${MODEL}")
    local LOG_DIR="/ssdwork/fuzhizhang/pns_results/${MODEL_NAME}"
    mkdir -p "${LOG_DIR}"
    local LOG="${LOG_DIR}/run.log"

    echo "[$(date '+%H:%M:%S')] Starting ${MODEL_NAME} on GPU ${GPU} → ${LOG}"

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} PNS_test/compute_pns.py \
        --model "${MODEL}" \
        ${COMMON_ARGS} \
        > "${LOG}" 2>&1

    echo "[$(date '+%H:%M:%S')] ✅ Finished ${MODEL_NAME} (GPU ${GPU})"
}

echo "════════════════════════════════════════════════════════════"
echo "  PNS 并行估算 - 6 个模型, 4 GPUs"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

# ── 第一批: 4 个模型并行 (GPU 0-3) ──
echo ""
echo "▶ 第一批: 4 个模型 (GPU 0,1,2,3)"
echo ""

run_model 0 "/home/fuzhizhang.fzz/model/Qwen2.5-1.5B-Base" &
PID0=$!
run_model 1 "/home/fuzhizhang.fzz/model/Qwen2.5-1.5B-Instruct" &
PID1=$!
run_model 2 "/home/fuzhizhang.fzz/model/DeepSeek-R1-Distill-Qwen-1.5B" &
PID2=$!
run_model 3 "/home/fuzhizhang.fzz/model/DeepScaleR-1.5B-Preview" &
PID3=$!

echo ""
echo "  等待第一批完成 (PID: ${PID0}, ${PID1}, ${PID2}, ${PID3})..."
wait ${PID0} ${PID1} ${PID2} ${PID3}
echo ""
echo "  ✅ 第一批全部完成!"

# ── 第二批: 2 个模型并行 (GPU 0-1) ──
echo ""
echo "▶ 第二批: 2 个模型 (GPU 0,1)"
echo ""

run_model 0 "/home/fuzhizhang.fzz/model/Qwen3-1.7B-Base" &
PID4=$!
run_model 1 "/home/fuzhizhang.fzz/model/Qwen3-1.7B" &
PID5=$!

echo ""
echo "  等待第二批完成 (PID: ${PID4}, ${PID5})..."
wait ${PID4} ${PID5}
echo ""
echo "  ✅ 第二批全部完成!"

# ── 汇总 ──
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  全部 6 个模型 PNS 估算完成! $(date)"
echo "  结果目录: /ssdwork/fuzhizhang/pns_results/"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "运行汇总对比:"
echo "  python PNS_test/compare_pns_results.py"
