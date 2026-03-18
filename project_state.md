# PNS_RLVR Project State

> **本文档是什么**: AI 助手（Cursor）与项目维护者共享的**唯一上下文来源**。所有关键路径、命令、实验结果、进行中任务都集中记录在此。
>
> **使用规则**:
> - AI 在回答任何问题前**必须先读此文档**，文档中已有的信息不再重复检索代码或历史对话。
> - 文档中找不到的信息，再去读源码/日志/nohup 文件。
> - 每完成子任务、发现 Bug、产出新结果时，**主动更新此文档**，覆盖旧信息，不做日记式追加。
>
> **项目一句话**: 基于 verl 框架的 DAPO 训练 + **PNS (Process Necessity Score)** 过程级奖励信号研究。在 Qwen3-1.7B-Base 上用 RLVR 训练数学推理能力，核心创新是通过 counterfactual ablation 给高 entropy 推理步骤打 PNS 分数并注入 advantage。

---

## 1. 静态与参考信息 (Reference Info)

### 模型与权重路径

| 用途 | 路径 |
|------|------|
| 基座模型 (训练用) | `/home/fuzhizhang.fzz/model/Qwen3-1.7B-Base` |
| R1蒸馏模型 (对照) | `/home/fuzhizhang.fzz/model/DeepSeek-R1-Distill-Qwen-1.5B` |
| 所有本地模型目录 | `/home/fuzhizhang.fzz/model/` (符号链接到 `/home/model/`) |
| DAPO-PNS merged ckpts | `/ssdwork/fuzhizhang/merged_models/DAPO-PNS-step{200,400,600,737,800,850}` |
| DAPO baseline merged ckpts | `/ssdwork/fuzhizhang/merged_models/DAPO-step{200,...,2550}` |
| Forking / Step-Forking ckpts | `/ssdwork/fuzhizhang/merged_models/{Forking,Step-Forking}-DAPO-step*` |
| DAPO-PNS 原始 ckpt (含optimizer) | `/ssdwork/fuzhizhang/ckpts/DAPO/RLVR-Qwen3-1.7B-DAPO-PNS/global_step_{200,400,600,737,800,850}` |

### 数据集与语料路径

| 数据集 | 路径 |
|--------|------|
| 训练集 (54.4k math) | `/home/fuzhizhang.fzz/data/math_combined_54k/math__combined_54.4k.parquet` |
| AIME 2024 | `/home/fuzhizhang.fzz/data/aime_2024/data/train-00000-of-00001.parquet` |
| AIME 2025 | `/home/fuzhizhang.fzz/data/aime25/test.jsonl` |
| MATH500 | `/home/fuzhizhang.fzz/data/math__math_500.parquet` |
| MATH500-Noop | `/home/fuzhizhang.fzz/CoT_Causal_Analysis/data/MATH500_noop/test.jsonl` |
| Val-AIME (32x重复) | `/home/fuzhizhang.fzz/data/math__aime_repeated_32x_960.parquet` |
| DeepScaleR-Preview-Dataset |  `/home/data/DeepScaleR-Preview-Dataset` |
| PNS 分析结果 | `/ssdwork/fuzhizhang/pns_results/` |

### 核心环境与依赖

| 项 | 值 |
|----|-----|
| Conda 环境 | `verl_fzz` → `/workspace/anaconda3/envs/verl_fzz/` |
| Python | 3.12.12 |
| PyTorch | 2.6.0+cu124 |
| vLLM | 0.8.5.post1 |
| CUDA Driver | 12.6 (H100 80GB, 机器名 `nv-h100-030`, 共8卡) |
| Ray 临时目录 | `/home/fuzhizhang.fzz/ray` (避免根分区磁盘满) |
| Flash-Attn whl | 项目根目录 `flash_attn-2.7.4.post1+cu12torch2.6*.whl` |

### 常用快捷命令

```bash
# 激活环境
conda activate verl_fzz

# === 训练 ===
# DAPO-PNS 真正干预 (GPU 4-7)
cd /home/fuzhizhang.fzz/PNS_RLVR && nohup bash recipe/rlvr_with_high_entropy_tokens_only/run_dapo_pns_qwen3_1_7B.sh > nohup.out.DAPO_PNS 2>&1 &

# DAPO-PNS dry-run 对照组 (GPU 4-7)
cd /home/fuzhizhang.fzz/PNS_RLVR && nohup bash recipe/rlvr_with_high_entropy_tokens_only/run_dapo_pns_dryrun_qwen3_1_7B.sh > nohup.out.DAPO_PNS_dry_run 2>&1 &

# === 评测 ===
# 单GPU评测 (修改 CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0 /workspace/anaconda3/envs/verl_fzz/bin/python tools/eval_benchmarks.py \
    --models /ssdwork/fuzhizhang/merged_models/DAPO-PNS-step400 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output /ssdwork/fuzhizhang/merged_models/eval_results.json

# === PNS 离线估算 (详细指令见下方 PNS 参数说明) ===
PY=/workspace/anaconda3/envs/verl_fzz/bin/python
MODEL=/ssdwork/fuzhizhang/merged_models/DAPO-PNS-step400  # 替换为目标模型

# 正向 PNS: 对正确解做 counterfactual ablation，测所有步骤
CUDA_VISIBLE_DEVICES=0 $PY PNS_test/compute_pns.py \
    --model $MODEL --benchmark math500 \
    --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
    --ablation_mode counterfactual \
    --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 \
    --test_all_steps --resume

# 反向 PNS: 对错误解做 counterfactual ablation，找 misleading 步骤
CUDA_VISIBLE_DEVICES=0 $PY PNS_test/compute_pns.py \
    --model $MODEL --benchmark math500 \
    --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
    --ablation_mode counterfactual \
    --test_all_steps --reverse --resume

# 多模型并行 (每个GPU一个模型)
nohup bash PNS_test/launch_parallel.sh > nohup.out.pns_parallel 2>&1 &

# PNS 结果汇总对比
$PY PNS_test/compare_pns_results.py

# 质检高PNS步骤的counterfactual rollout
CUDA_VISIBLE_DEVICES=0 $PY PNS_test/inspect_pns_rollouts.py \
    --model $MODEL --results_dir /ssdwork/fuzhizhang/pns_results/$(basename $MODEL) \
    --num_examples 5 --num_rollouts 2

# === 监控 ===
nvidia-smi                            # GPU 使用情况
tail -f nohup.out.0316.DAPO_PNS_dry_run  # 跟踪训练日志
ps aux | grep main_dapo               # 查看训练进程
```

### PNS 离线估算参数说明 (`PNS_test/compute_pns.py`)

**4 阶段 Pipeline**: Phase1 生成正确/错误解 → Phase2 HF模型算 per-token entropy + 步骤分割 → Phase3 vLLM ablation rollout → Phase4 统计分析

**核心参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必填) | merged HF checkpoint 路径 |
| `--benchmark` | `math500` | 数据集，可选 `math500` / `aime24` |
| `--num_problems` | 50 | 取多少题（0=全部） |
| `--num_initial_samples` | 16 | Phase1 每题生成多少个样本来找正确/错误解 |
| `--num_rollouts` | 5 | Phase3 每个 ablation 任务的 rollout 数 |
| `--max_tokens` | 16384 | 生成最大 token 数 |
| `--seed` | 42 | 随机种子 |
| `--resume` | - | 断点续跑，跳过已完成的 phase |

**关键选择参数**:

| 参数 | 选项 | 当前选择 | 说明 |
|------|------|---------|------|
| `--ablation_mode` | `counterfactual` / `truncate` / `skip` | **counterfactual** | **counterfactual**: 保留前文 A，告诉模型"不要用步骤 B，找替代方案"（原版 PNS 论文方法，最接近因果推断）。**truncate**: 截断到步骤 B 之前，从头续写（测"B 是不是 pivot 点"）。**skip**: 删掉步骤 B 但保留后续步骤，让模型重新推答案（测"B 在其他推理都在的情况下是否必要"）。 |
| `--test_all_steps` | flag | **建议开启** | 不加此 flag 时只测 top 20% 高 entropy 步骤（更快，但无法算 HE vs LE 对比和 entropy↔PNS 相关性）。加了后测所有步骤（慢 ~5x，但能得到完整的 entropy↔PNS 相关性、HE/LE 对比、position 分析）。 |
| `--reverse` | flag | 按需 | 正向模式：对正确解 ablation，PNS = 步骤被移除后答对率下降多少。反向模式：对错误解 ablation，reverse_PNS = 步骤被移除后答对率上升多少（找"误导性"步骤）。 |
| `--entropy_top_ratio` | float | 0.2 | token 级别：取 entropy 最高的 top 20% token 来给步骤打分 |
| `--step_entropy_top_ratio` | float | 0.2 | step 级别：取 entropy score 最高的 top 20% 步骤标记为高 entropy |

**输出文件** (自动存到 `/ssdwork/fuzhizhang/pns_results/{model_name}/`):

| 文件 | 内容 |
|------|------|
| `phase1_correct_responses.json` | 找到的正确解（含 response_token_ids） |
| `phase2_entropy_steps.json` | 步骤分割 + per-token entropy + 高 entropy 标记 |
| `phase3_ablation_rollouts.json` | 每个步骤的 ablation rollout 结果 + PNS 值 |
| `pns_analysis.json` | 汇总统计（mean_PNS, 分布, HE vs LE, position, top20 步骤等） |
| `reverse_*.json` | 反向模式的对应文件（`--reverse` 时生成） |

**已有实验的标准配置**: `--benchmark math500 --num_problems 50 --num_rollouts 5 --ablation_mode counterfactual --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --test_all_steps --resume`

### 关键脚本位置

| 脚本 | 路径 |
|------|------|
| DAPO-PNS 训练 | `recipe/rlvr_with_high_entropy_tokens_only/run_dapo_pns_qwen3_1_7B.sh` |
| DAPO-PNS dry-run | `recipe/rlvr_with_high_entropy_tokens_only/run_dapo_pns_dryrun_qwen3_1_7B.sh` |
| DAPO baseline (纯entropy forking) | `recipe/rlvr_with_high_entropy_tokens_only/run_dapo_qwen3_1_7B.sh` |
| R1-Distill-1.5B baseline | `recipe/rlvr_with_high_entropy_tokens_only/run_dapo_r1_distill_1_5B.sh` |
| Benchmark 评测脚本 | `tools/eval_benchmarks.py` |
| PNS 训练核心逻辑 | `verl/trainer/ppo/pns_utils.py` |
| DAPO trainer | `recipe/dapo/dapo_ray_trainer.py` |
| **PNS 离线估算** (正向+反向) | `PNS_test/compute_pns.py` |
| PNS 结果对比汇总 | `PNS_test/compare_pns_results.py` |
| PNS rollout 质检工具 | `PNS_test/inspect_pns_rollouts.py` |
| PNS 并行启动 (6模型) | `PNS_test/launch_parallel.sh` |
| PNS 串行启动 (6模型) | `PNS_test/run_pns_6models.sh` |

---

## 2. 动态任务状态 (Dynamic Tasks)

### 当前进行中 (In Progress)

- **DAPO-PNS dry-run 训练**: 在 GPU 4-7 上运行 `RLVR-Qwen3-1.7B-DAPO-PNS-dryrun`。日志: `nohup.out.0316.DAPO_PNS_dry_run`。每步约 5min（含 PNS ablation ~2.5min）。

### 待办事项 (To-Do)

1. 等待 DAPO-PNS dry-run 训练完成，merge checkpoints 并评测
2. 对比 PNS 干预组 vs dry-run 组的训练曲线（wandb 项目: DAPO）
3. R1-Distill-1.5B baseline 尚未启动训练

### 已完成与中间结果 (Completed & Results)

#### DAPO-PNS 训练 (干预组) — 已完成 850 steps

Checkpoint 保存于 `/ssdwork/fuzhizhang/ckpts/DAPO/RLVR-Qwen3-1.7B-DAPO-PNS/`，已 merge 到 `/ssdwork/fuzhizhang/merged_models/DAPO-PNS-step*`。

#### DAPO-PNS 评测结果 (Greedy Acc@1 %)

| Step | AIME24 | AIME25 | MATH500 | MATH500-Noop |
|------|--------|--------|---------|--------------|
| 200  | 6.7    | 3.3    | 64.6    | 56.2         |
| 400  | 3.3    | 3.3    | **67.0**| **59.6**     |
| 600  | 6.7    | 0.0    | 63.6    | 54.0         |
| 737  | **10.0**| **6.7**| 63.4    | 43.8         |
| 800  | 10.0   | 6.7    | 65.8    | 48.0         |
| 850  | 3.3    | 3.3    | 67.8    | 49.8         |

> Best MATH500: step400 (67.0%) / step850 (67.8%); Best AIME: step737 (10.0/6.7%)
> 训练后期 response length 暴涨，MATH500-Noop 下降（过拟合迹象）

**DAPO-PNS 平均生成长度 (Len@1 greedy / Len@16 sampling)**

| Step | AIME24 | AIME25 | MATH500 | Noop | AIME24 | AIME25 | MATH500 | Noop |
|------|--------|--------|---------|------|--------|--------|---------|------|
|      | Len@1  | Len@1  | Len@1   | Len@1| Len@16 | Len@16 | Len@16  | Len@16|
| 200  | 5565   | 5092   | 1653    | 1897 | 1312   | 1137   | 703     | 781  |
| 400  | 5676   | 3613   | 1521    | 2280 | 1188   | 1068   | 663     | 728  |
| 600  | 9762   | 8306   | 2156    | 2895 | 1429   | 1317   | 719     | 834  |
| 737  | 11478  | 11733  | 3587    | 6699 | 3728   | 3427   | 1939    | 2352 |
| 800  | 7969   | 9101   | 2927    | 4672 | 1242   | 1273   | 991     | 1030 |
| 850  | 11699  | 8530   | 3095    | 5051 | 1362   | 1427   | 1064    | 1120 |

> step737 长度暴涨最严重（MATH500 Len@1 3587，Noop 6699）；step400 最简洁（MATH500 Len@1 1521）。

#### Baseline 评测结果 — 已完成 (2025-03-18)

**Greedy Acc@1 (%)**

| Model | AIME24 | AIME25 | MATH500 | MATH500-Noop |
|-------|--------|--------|---------|--------------|
| Qwen2.5-1.5B-Base | 0.0 | 0.0 | 0.2 | 0.6 |
| Qwen2.5-1.5B-Instruct | 3.3 | 0.0 | 53.0 | 46.4 |
| Qwen3-1.7B-Base (基座) | 3.3 | 13.3 | 56.8 | 48.2 |
| DeepSeek-R1-Distill-1.5B | 16.7 | 13.3 | 68.4 | 35.4 |
| DeepScaleR-1.5B-Preview | **36.7** | 26.7 | 81.2 | 49.8 |
| **Qwen3-1.7B** (蒸馏SFT) | 33.3 | 23.3 | **84.6** | **69.8** |

**Pass@16 (%)**

| Model | AIME24 | AIME25 | MATH500 | MATH500-Noop |
|-------|--------|--------|---------|--------------|
| Qwen2.5-1.5B-Base | 3.3 | 0.0 | 18.6 | 17.0 |
| Qwen2.5-1.5B-Instruct | 10.0 | 6.7 | 81.2 | 73.6 |
| Qwen3-1.7B-Base (基座) | 10.0 | 16.7 | 78.6 | 73.2 |
| DeepSeek-R1-Distill-1.5B | 66.7 | 50.0 | 94.2 | 90.0 |
| DeepScaleR-1.5B-Preview | 50.0 | 46.7 | 94.6 | 89.0 |
| **Qwen3-1.7B** (蒸馏SFT) | **70.0** | 46.7 | **96.0** | 88.0 |

**平均生成长度 (Len@1 greedy / Len@16 sampling)**

| Model | AIME24 | AIME25 | MATH500 | Noop | AIME24 | AIME25 | MATH500 | Noop |
|-------|--------|--------|---------|------|--------|--------|---------|------|
|       | Len@1  | Len@1  | Len@1   | Len@1| Len@16 | Len@16 | Len@16  | Len@16|
| Qwen2.5-1.5B-Base | 15841 | 15843 | 15737 | 15162 | 2636 | 2525 | 2939 | 2697 |
| Qwen2.5-1.5B-Instruct | 2825 | 5413 | 1210 | 1444 | 762 | 782 | 515 | 541 |
| Qwen3-1.7B-Base | 6636 | 5902 | 1412 | 1685 | 1226 | 1325 | 790 | 845 |
| R1-Distill-1.5B | 14199 | 14266 | 6199 | 10986 | 11537 | 11081 | 4820 | 6554 |
| DeepScaleR-1.5B | 10369 | 12747 | 4010 | 8400 | 9246 | 8634 | 3654 | 5003 |
| Qwen3-1.7B | 12720 | 13426 | 5306 | 7306 | 13419 | 13234 | 5162 | 7070 |

> **Qwen3-1.7B (蒸馏SFT，由235B/32B RL模型数据蒸馏) 是当前全方位最强 baseline**：MATH500 84.6%, MATH500-Noop 69.8%（鲁棒性远超 DeepScaleR 的 49.8%）。Pass@16 在 AIME24 70.0% 和 MATH500 96.0% 也领先。
> DeepScaleR AIME24 在两次评测中有较大方差 (36.7% vs 26.7%)，因 AIME 仅 30 题，vLLM greedy 在不同 GPU/batch 配置下存在浮点非确定性。
> R1-Distill 的 MATH500-Noop 严重退化 (68.4→35.4)，Qwen3-1.7B 最鲁棒 (84.6→69.8，仅降 14.8pp)。

#### PNS 离线估算结果 (Forward & Reverse)

**工具**: `PNS_test/compute_pns.py` (4 阶段: 生成正确解→entropy计算→ablation rollout→分析)
**输出目录**: `/ssdwork/fuzhizhang/pns_results/{model_name}/`
**Benchmark**: MATH500, 50 题, 5 rollouts, counterfactual ablation, test_all_steps

**正向 PNS (Forward)**: 对**正确**解的步骤做 ablation，PNS = 1 - (ablation后正确率)。高 PNS = 步骤不可或缺。

| Model | Solved | Steps | mean_PNS | Ent↔PNS corr | HE_PNS | LE_PNS | Δ(HE-LE) |
|-------|--------|-------|----------|---------------|--------|--------|----------|
| Qwen3-1.7B-Base | 38 | 950 | 0.5661 | 0.150 | 0.6154 | 0.5462 | +0.069 |
| DAPO-PNS-step200 | 41 | 1030 | 0.3324 | 0.082 | 0.3210 | 0.3373 | -0.016 |
| DAPO-PNS-step400 | 42 | 1366 | 0.3275 | 0.128 | 0.3143 | 0.3330 | -0.019 |
| DAPO-PNS-step600 | 41 | 2040 | 0.5664 | -0.002 | 0.5361 | 0.5782 | -0.042 |
| DAPO-PNS-step800 | 41 | 2263 | 0.3949 | 0.051 | 0.3770 | 0.4020 | -0.025 |
| DeepScaleR-1.5B | 48 | 9782 | 0.2270 | 0.127 | 0.2194 | 0.2292 | -0.010 |

**反向 PNS (Reverse)**: 对**错误**解的步骤做 ablation，reverse_PNS = (ablation后正确率)。高值 = 步骤是导致错误的元凶。

| Model | Solved(错误) | Steps | mean_reverse_PNS |
|-------|-------------|-------|------------------|
| Qwen3-1.7B-Base | 50 | 4458 | 0.0626 |
| DAPO-PNS-step200 | 37 | 1486 | 0.2096 |
| DAPO-PNS-step400 | 38 | 1555 | 0.2385 |
| DAPO-PNS-step600 | 37 | 2033 | 0.2003 |
| DAPO-PNS-step800 | 44 | 2826 | 0.2828 |
| DeepScaleR-1.5B | 19 | 9113 | 0.3328 |

> **发现**: Entropy↔PNS 相关性弱 (corr < 0.15)，PNS 提供了 entropy 无法捕获的新信号。
> 训练中期步数暴涨但 mean_PNS 未稳定提升；DeepScaleR 步多但 PNS 低（冗余步骤多）。

**PNS 结果已完成的模型**: Qwen2.5-1.5B-Base, Qwen2.5-1.5B-Instruct, DeepSeek-R1-Distill-1.5B, DeepScaleR-1.5B, Qwen3-1.7B-Base, Qwen3-1.7B, DAPO-PNS-step{200,400,600,800}(allsteps+reverse), DAPO-PNS-step{737,850}(forward only)

#### 关键 Bug 记录

- `compute_score` (sympy.simplify) 可能死循环挂起 → 已在 `eval_benchmarks.py` 中用 `multiprocessing` 超时 15s 包装（`compute_score_safe`）
- Ray gRPC 超时导致 actor unavailable → 已在训练脚本中设置 `RAY_grpc_keepalive_timeout_ms=900000` 等环境变量
- 根分区磁盘满 → Ray 临时目录已改到 `/home/fuzhizhang.fzz/ray`

#### 评测结果文件

| 文件 | 内容 |
|------|------|
| `/ssdwork/fuzhizhang/merged_models/eval_results.json` | DAPO-PNS step{200-850} 全指标 |
| `/ssdwork/fuzhizhang/merged_models/eval_results_baselines.json` | 全部 6 个 baseline 模型（含 Qwen3-1.7B 蒸馏SFT）Acc@1 + Pass@16 |
| `/ssdwork/fuzhizhang/merged_models/eval_results_baselines_gpu0.json` | 早期 GPU0 评测（DeepScaleR, Qwen3-1.7B），数值与 baselines.json 有小差异 |
| `/ssdwork/fuzhizhang/eval_results_all.json` | 早期 DAPO/Forking/Step-Forking 全量结果 |
