#!/usr/bin/env bash
#
# PNS 估算: 6 个预训练/微调模型的对比实验
#
# 模型列表 (2 个系列):
#   Qwen2.5-1.5B 系列:
#     1. Qwen2.5-1.5B-Base            (base预训练)
#     2. Qwen2.5-1.5B-Instruct        (SFT)
#     3. DeepSeek-R1-Distill-Qwen-1.5B (蒸馏)
#     4. DeepScaleR-1.5B-Preview       (蒸馏+RL)
#
#   Qwen3-1.7B 系列:
#     5. Qwen3-1.7B-Base              (base预训练)
#     6. Qwen3-1.7B                   (RL)
#
# 实验目标:
#   - 观察不同训练阶段 (Base→SFT→Distill→RL) 的 PNS 分布变化
#   - 分析 entropy↔PNS 相关性在不同模型上是否一致
#   - 为后续 PNS-DAPO 训练提供 baseline 估计
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash PNS_test/run_pns_6models.sh
#
# 如需指定 GPU:
#   CUDA_VISIBLE_DEVICES=3 bash PNS_test/run_pns_6models.sh

set -euo pipefail

cd /home/fuzhizhang.fzz/PNS_RLVR

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
BENCHMARK="math500"
NUM_PROBLEMS=50
NUM_ROLLOUTS=5
ENTROPY_TOP_RATIO=0.2
STEP_ENTROPY_TOP_RATIO=0.2

# 是否测试所有步骤 (用于 entropy↔PNS 相关性分析)
# 如果设为 true，运行时间会更长，但能得到 entropy↔PNS 相关性
TEST_ALL_STEPS=true

# ──────────────────────────────────────────────────────────────
# 模型列表
# ──────────────────────────────────────────────────────────────
MODELS=(
    # Qwen2.5-1.5B 系列
    "/home/fuzhizhang.fzz/model/Qwen2.5-1.5B-Base"
    "/home/fuzhizhang.fzz/model/Qwen2.5-1.5B-Instruct"
    "/home/fuzhizhang.fzz/model/DeepSeek-R1-Distill-Qwen-1.5B"
    "/home/fuzhizhang.fzz/model/DeepScaleR-1.5B-Preview"
    # Qwen3-1.7B 系列
    "/home/fuzhizhang.fzz/model/Qwen3-1.7B-Base"
    "/home/fuzhizhang.fzz/model/Qwen3-1.7B"
)

# ──────────────────────────────────────────────────────────────
# 逐模型运行
# ──────────────────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
SUCCESS=0
FAILED=0

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME=$(basename "${MODEL}")
    IDX=$((i + 1))

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  [${IDX}/${TOTAL}] ${MODEL_NAME}"
    echo "╚══════════════════════════════════════════════════════════╝"

    # 检查模型路径是否存在
    if [ ! -d "${MODEL}" ]; then
        echo "  ⚠️  模型路径不存在: ${MODEL}, 跳过"
        FAILED=$((FAILED + 1))
        continue
    fi

    EXTRA_FLAGS=""
    if [ "${TEST_ALL_STEPS}" = "true" ]; then
        EXTRA_FLAGS="--test_all_steps"
    fi

    python PNS_test/compute_pns.py \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --num_problems ${NUM_PROBLEMS} \
        --num_rollouts ${NUM_ROLLOUTS} \
        --entropy_top_ratio ${ENTROPY_TOP_RATIO} \
        --step_entropy_top_ratio ${STEP_ENTROPY_TOP_RATIO} \
        ${EXTRA_FLAGS} \
        --resume \
    && SUCCESS=$((SUCCESS + 1)) \
    || { echo "  ❌ Failed for ${MODEL_NAME}"; FAILED=$((FAILED + 1)); }

    echo ""
    echo "  ✅ Completed ${MODEL_NAME}"
    echo "  Results: /home/fuzhizhang.fzz/model/merge_models/pns_results/${MODEL_NAME}/"
    echo ""
done

# ──────────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PNS 估算完成"
echo "  成功: ${SUCCESS}/${TOTAL}  失败: ${FAILED}/${TOTAL}"
echo "  结果目录: /home/fuzhizhang.fzz/model/merge_models/pns_results/"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "后续步骤: 运行汇总对比脚本"
echo "  python PNS_test/compare_pns_results.py"
