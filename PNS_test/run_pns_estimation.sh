#!/usr/bin/env bash
#
# PNS Estimation Runner
#
# 用法说明:
#   1. 基础运行: 在某个模型 checkpoint 上运行 PNS 估算
#   2. 对比运行: 在多个模型上分别运行, 对比 PNS 分布
#   3. 全步骤分析: --test_all_steps 模式可以分析 entropy↔PNS 相关性
#
# 结果保存在 /ssdwork/fuzhizhang/pns_results/{model_name}/
#   - phase1_correct_responses.json  (正确回答)
#   - phase2_entropy_steps.json      (熵 + 步骤分析)
#   - phase3_ablation_rollouts.json  (消融 rollout 结果)
#   - pns_analysis.json              (PNS 统计分析)

set -xeuo pipefail

cd /home/fuzhizhang.fzz/PNS_RLVR

# ──────────────────────────────────────────────────────────────
# 可选的模型路径 (取消注释你想运行的模型)
# ──────────────────────────────────────────────────────────────

# 基线模型
# MODEL=/ssdwork/fuzhizhang/merged_models/Qwen3-1.7B-Base

# DAPO 训练后模型 (不同 step)
# MODEL=/ssdwork/fuzhizhang/merged_models/DAPO-step400
# MODEL=/ssdwork/fuzhizhang/merged_models/DAPO-step800
# MODEL=/ssdwork/fuzhizhang/merged_models/DAPO-step1600

# Forking DAPO (token-level 高熵选择)
# MODEL=/ssdwork/fuzhizhang/merged_models/Forking-DAPO-step400
# MODEL=/ssdwork/fuzhizhang/merged_models/Forking-DAPO-step800
# MODEL=/ssdwork/fuzhizhang/merged_models/Forking-DAPO-step1600

# Step Forking DAPO (step-level 高熵选择)
MODEL=/ssdwork/fuzhizhang/merged_models/Step-Forking-DAPO-step1600

# ──────────────────────────────────────────────────────────────
# 运行模式选择
# ──────────────────────────────────────────────────────────────

# 模式 1: 快速测试 (仅高熵步骤, 少量问题)
# 适合快速验证管线是否正常工作
# CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
#     --model ${MODEL} \
#     --benchmark math500 \
#     --num_problems 20 \
#     --num_rollouts 5 \
#     --entropy_top_ratio 0.2 \
#     --step_entropy_top_ratio 0.2

# 模式 2: 标准运行 (仅高熵步骤, 50 题)
# 这是推荐的首次运行配置
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model ${MODEL} \
    --benchmark math500 \
    --num_problems 50 \
    --num_rollouts 5 \
    --entropy_top_ratio 0.2 \
    --step_entropy_top_ratio 0.2

# 模式 3: 全步骤分析 (所有步骤, 用于 entropy↔PNS 相关性分析)
# 注意: 运行时间会较长 (所有步骤 × 5 rollouts)
# CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
#     --model ${MODEL} \
#     --benchmark math500 \
#     --num_problems 30 \
#     --num_rollouts 5 \
#     --test_all_steps \
#     --entropy_top_ratio 0.2 \
#     --step_entropy_top_ratio 0.2

# 模式 4: Skip 模式消融 (从中间去掉步骤, 保留后续步骤)
# CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
#     --model ${MODEL} \
#     --benchmark math500 \
#     --num_problems 50 \
#     --num_rollouts 5 \
#     --ablation_mode skip \
#     --entropy_top_ratio 0.2 \
#     --step_entropy_top_ratio 0.2

# ──────────────────────────────────────────────────────────────
# 批量对比运行 (取消注释运行)
# 在多个模型上分别计算 PNS, 用于对比不同训练方法
# ──────────────────────────────────────────────────────────────
# for MODEL in \
#     /ssdwork/fuzhizhang/merged_models/Qwen3-1.7B-Base \
#     /ssdwork/fuzhizhang/merged_models/DAPO-step1600 \
#     /ssdwork/fuzhizhang/merged_models/Forking-DAPO-step1600 \
#     /ssdwork/fuzhizhang/merged_models/Step-Forking-DAPO-step1600; do
#
#     echo "========================================"
#     echo "Running PNS for: $(basename ${MODEL})"
#     echo "========================================"
#
#     CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
#         --model ${MODEL} \
#         --benchmark math500 \
#         --num_problems 50 \
#         --num_rollouts 5 \
#         --test_all_steps \
#         --resume
# done
