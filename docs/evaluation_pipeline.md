# Evaluation Pipeline

本文档描述 PNS_RLVR 项目中模型评估的完整流程，包含 3 个阶段：
1. **Merge**: 将 FSDP 分布式 checkpoint 合并为 HuggingFace 格式
2. **Benchmark Eval**: 在 AIME24/AIME25/MATH500/MATH500_Noop 上评估
3. **PNS Estimation**: 估算推理步骤的 Process Necessity Score

---

## 前置条件

- Conda 环境: `verl_fzz` (路径: `/workspace/anaconda3/envs/verl_fzz`)
- 工作目录: `/home/fuzhizhang.fzz/PNS_RLVR`
- 至少 1 张空闲 GPU (H100 80GB)

```bash
conda activate verl_fzz
cd /home/fuzhizhang.fzz/PNS_RLVR
```

---

## 阶段 1: Merge FSDP Checkpoint

### 原理

verl 训练框架使用 FSDP (Fully Sharded Data Parallel) 保存 checkpoint，模型参数被 shard 到多个 rank 文件中。Merge 步骤将这些 shard 重新组合为标准 HuggingFace 格式 (model.safetensors + config.json + tokenizer)。

### Checkpoint 目录结构

```
{ckpt_root}/global_step_{N}/
  actor/
    fsdp_config.json              # FSDP 配置 (含 world_size)
    model_world_size_4_rank_0.pt  # 模型 shard (rank 0)
    model_world_size_4_rank_1.pt  # 模型 shard (rank 1)
    model_world_size_4_rank_2.pt  # 模型 shard (rank 2)
    model_world_size_4_rank_3.pt  # 模型 shard (rank 3)
    huggingface/                  # HF 配置文件 (config.json, tokenizer 等)
    optim_world_size_4_rank_*.pt  # 优化器状态 (merge 不需要)
    extra_state_world_size_4_rank_*.pt  # 额外状态 (merge 不需要)
  data.pt                         # 训练数据状态
```

### 命令

**单个 checkpoint merge:**
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir {ckpt_root}/global_step_{N}/actor \
    --target_dir {output_dir}/{model_name}
```

**批量 merge 示例:**
```bash
CKPT_ROOT="/ssdwork/fuzhizhang/ckpts/DAPO/RLVR-Qwen3-1.7B-DAPO-PNS"
MERGE_DIR="/ssdwork/fuzhizhang/merged_models"
PYTHON="/workspace/anaconda3/envs/verl_fzz/bin/python"

for STEP in 200 400 600 737 800 850; do
    ${PYTHON} -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CKPT_ROOT}/global_step_${STEP}/actor" \
        --target_dir "${MERGE_DIR}/DAPO-PNS-step${STEP}"
done
```

### 输出

merge 后的目录结构:
```
{output_dir}/{model_name}/
  config.json
  model.safetensors          # 合并后的完整模型权重
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  vocab.json / merges.txt    # 视模型而定
```

### 注意事项

- Merge 过程主要在 CPU 上完成，不需要 GPU
- 1.7B 模型 merge 约需 10-15 秒/checkpoint
- Merge 前会检查 `model.safetensors` 或 `config.json` 是否已存在，已 merge 的会自动跳过

### 相关代码

| 文件 | 说明 |
|------|------|
| `verl/model_merger/__main__.py` | CLI 入口 (推荐使用) |
| `verl/model_merger/base_model_merger.py` | 基类 + 配置解析 |
| `verl/model_merger/fsdp_model_merger.py` | FSDP 后端 merge 实现 |
| `scripts/legacy_model_merger.py` | 旧版独立 merger (不推荐) |

---

## 阶段 2: Benchmark 评估

### 支持的 Benchmark

| Benchmark | 数据路径 | 题目数 | 说明 |
|-----------|----------|--------|------|
| `aime24` | `/home/fuzhizhang.fzz/data/aime_2024/data/train-00000-of-00001.parquet` | 30 | AIME 2024 竞赛题 |
| `aime25` | `/home/fuzhizhang.fzz/data/aime25/test.jsonl` | 30 | AIME 2025 竞赛题 |
| `math500` | `/home/fuzhizhang.fzz/data/math__math_500.parquet` | 500 | MATH benchmark 子集 |
| `math500_noop` | `/home/fuzhizhang.fzz/CoT_Causal_Analysis/data/MATH500_noop/test.jsonl` | 500 | MATH500 No-Op 变体 |

### 评估指标

每个 benchmark 生成 5 个指标:

| 指标 | 生成设置 | 含义 |
|------|----------|------|
| `acc_at_1` | temp=0, n=1 (greedy) | 贪心准确率 (%) |
| `len_at_1` | temp=0, n=1 (greedy) | 贪心平均生成长度 (tokens) |
| `acc_at_16` | temp=1.0, n=16 (sampling) | 16 次采样的平均准确率 (%) |
| `pass_at_16` | temp=1.0, n=16 (sampling) | 16 次采样中至少 1 次正确的比例 (%) |
| `len_at_16` | temp=1.0, n=16 (sampling) | 16 次采样的平均生成长度 (tokens) |

### 命令

**评估单个模型:**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
    --models /path/to/merged_model \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output /path/to/eval_results.json
```

**评估多个模型 (串行，同一 GPU):**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
    --models \
        /path/to/model1 \
        /path/to/model2 \
        /path/to/model3 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output /path/to/eval_results.json
```

**多 GPU 并行评估 (推荐):**

将模型分组到不同 GPU 上，使用不同的 output 文件，最后合并:

```bash
MERGE_DIR="/ssdwork/fuzhizhang/merged_models"

# GPU 0: 前半
CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
    --models ${MERGE_DIR}/model1 ${MERGE_DIR}/model2 ${MERGE_DIR}/model3 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output ${MERGE_DIR}/eval_results_gpu0.json &

# GPU 1: 后半
CUDA_VISIBLE_DEVICES=1 python tools/eval_benchmarks.py \
    --models ${MERGE_DIR}/model4 ${MERGE_DIR}/model5 ${MERGE_DIR}/model6 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output ${MERGE_DIR}/eval_results_gpu1.json &

wait

# 合并结果
python -c "
import json
r0 = json.load(open('${MERGE_DIR}/eval_results_gpu0.json'))
r1 = json.load(open('${MERGE_DIR}/eval_results_gpu1.json'))
r0.update(r1)
with open('${MERGE_DIR}/eval_results.json', 'w') as f:
    json.dump(r0, f, indent=2)
"
```

### 断点续传

eval_benchmarks.py 支持增量评估:
- 结果会在每个 benchmark 完成后保存到 output 文件
- 重新运行时会自动跳过已完成的 model+benchmark 组合
- 如果中途中断，重新运行相同命令即可

### 查看结果

```bash
# JSON 格式
cat /path/to/eval_results.json | python -m json.tool

# 结果结构示例:
# {
#   "DAPO-PNS-step200": {
#     "aime24": {"acc_at_1": 6.7, "len_at_1": 3500, "acc_at_16": 8.5, "pass_at_16": 33.3, "len_at_16": 3800},
#     "math500": {...},
#     ...
#   },
#   "DAPO-PNS-step400": {...},
#   ...
# }
```

### 注意事项

- 每个模型需要约 4GB GPU 显存 (1.7B model)，但 vLLM KV cache 会占用大量显存 (总共约 70GB on H100)
- `max_tokens=16384` 对应 `max_model_len=18432` (16384 + 2048 buffer)
- sampling@16 (n=16, temp=1.0) 比 greedy 慢约 16 倍
- MATH500 的 sampling@16 是最耗时的部分 (500 题 x 16 次采样)
- 评分使用 `verl.utils.reward_score.naive_dapo.compute_score`，带 60 秒超时保护 (sympy 可能挂起)

### 相关代码

| 文件 | 说明 |
|------|------|
| `tools/eval_benchmarks.py` | Benchmark 评估主脚本 |
| `verl/utils/reward_score/naive_dapo.py` | 数学答案评分函数 |

---

## 阶段 3: PNS (Process Necessity Score) 估算

### 原理

PNS 衡量推理过程中每个步骤的必要性:
1. 生成正确解答，分割为推理步骤
2. 计算每个 token 的熵，选出高熵步骤
3. 对每个目标步骤做消融 (ablation): 移除该步骤后让模型重新 rollout N 次
4. `PNS = 1 - (ablation 后正确次数 / N)`

PNS 值越高 = 该步骤越不可或缺。

### 4 个阶段 (Phase)

| Phase | 说明 | 使用 |
|-------|------|------|
| Phase 1 | 生成正确解答 (vLLM, temp=1.0, n=16) | GPU (vLLM) |
| Phase 2 | 计算 token 熵 + 步骤分割 + 选择高熵步骤 | GPU (HF model) |
| Phase 3 | 消融 rollout (vLLM) | GPU (vLLM) |
| Phase 4 | 统计分析 + 输出报告 | CPU |

### 命令

**标准运行 (推荐首次运行):**
```bash
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model /path/to/merged_model \
    --benchmark math500 \
    --num_problems 50 \
    --num_rollouts 5 \
    --entropy_top_ratio 0.2 \
    --step_entropy_top_ratio 0.2
```

**全步骤分析 (用于 entropy-PNS 相关性研究):**
```bash
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model /path/to/merged_model \
    --benchmark math500 \
    --num_problems 50 \
    --num_rollouts 5 \
    --test_all_steps \
    --entropy_top_ratio 0.2 \
    --step_entropy_top_ratio 0.2
```

**断点续传 (中断后继续):**
```bash
# 加 --resume 会跳过已完成的 phase
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model /path/to/merged_model \
    --resume
```

**反向 PNS (分析错误路径):**
```bash
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model /path/to/merged_model \
    --reverse \
    --resume
```

### 消融模式

| 模式 | `--ablation_mode` | 说明 |
|------|-------------------|------|
| Counterfactual | `counterfactual` (默认) | 告诉模型 "不要使用这个步骤"，让它找替代方案 |
| Truncate | `truncate` | 截断到目标步骤之前，让模型重新生成 |
| Skip | `skip` | 从中间移除目标步骤，保留其后步骤 |

### PNS 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_problems` | 50 | 使用的题目数 |
| `--num_initial_samples` | 16 | 每题采样次数 (用于找到正确解答) |
| `--num_rollouts` | 5 | 每步消融后的 rollout 次数 |
| `--entropy_top_ratio` | 0.2 | Token 级别高熵比例 (top 20% tokens) |
| `--step_entropy_top_ratio` | 0.2 | Step 级别高熵比例 (top 20% steps) |
| `--max_tokens` | 16384 | 最大生成 token 数 |
| `--seed` | 42 | 随机种子 |

### 输出文件

结果保存在 `/ssdwork/fuzhizhang/pns_results/{model_name}/`:

| 文件 | 说明 |
|------|------|
| `config.json` | 运行配置 |
| `phase1_correct_responses.json` | 正确解答 |
| `phase2_entropy_steps.json` | 熵分析 + 步骤分割 |
| `phase3_ablation_rollouts.json` | 消融 rollout 结果 |
| `pns_analysis.json` | PNS 统计报告 |

### 多模型 PNS 对比

```bash
python PNS_test/compare_pns_results.py \
    --results_dir /ssdwork/fuzhizhang/pns_results
```

### 相关代码

| 文件 | 说明 |
|------|------|
| `PNS_test/compute_pns.py` | PNS 估算主脚本 |
| `PNS_test/compare_pns_results.py` | 多模型 PNS 对比 |
| `PNS_test/run_pns_estimation.sh` | 单模型 PNS 运行模板 |
| `PNS_test/run_pns_6models.sh` | 6 模型串行 PNS |
| `PNS_test/launch_parallel.sh` | 多 GPU 并行 PNS |

---

## 一键运行脚本

### 全流程 (merge + eval + PNS)

```bash
bash scripts/merge_and_eval_dapo_pns.sh
```

### 环境变量控制

| 变量 | 默认 | 说明 |
|------|------|------|
| `SKIP_MERGE` | 0 | 设为 1 跳过 merge |
| `SKIP_EVAL` | 0 | 设为 1 跳过 eval |
| `SKIP_PNS` | 0 | 设为 1 跳过 PNS |
| `EVAL_GPU` | 0 | 指定 GPU 编号 |

```bash
# 示例: 只跑 eval，使用 GPU 1
SKIP_MERGE=1 SKIP_PNS=1 EVAL_GPU=1 bash scripts/merge_and_eval_dapo_pns.sh
```

### 双 GPU 并行 (推荐)

```bash
# 并行 eval (GPU 0+1) + 并行 PNS (GPU 0+1)
bash scripts/parallel_eval_dapo_pns.sh
```

---

## 常见问题

### Q: vLLM 报 "No available memory for the cache blocks"
A: GPU 显存不足。可能原因:
1. 之前 kill 的进程残留了 CUDA 内存 -> 用 `nvidia-smi` 查看并 `kill -9` 残留进程
2. 其他进程占用了 GPU -> 用 `CUDA_VISIBLE_DEVICES` 切换到空闲 GPU
3. 模型太大 -> 降低 `gpu_memory_utilization` (代码中默认 0.90)

### Q: compute_score 超时
A: sympy.simplify() 可能在某些表达式上无限挂起。eval_benchmarks.py 中已包含 60 秒超时机制 (使用 multiprocessing)，超时的会标记为错误。

### Q: Merge 后模型文件没有 model.safetensors
A: 检查 checkpoint 的 actor 目录下是否有 `huggingface/config.json`。新版 merger (`python -m verl.model_merger`) 会自动在 `{local_dir}/huggingface` 找配置。

### Q: 想评估新的 checkpoint
A: 只需修改脚本中的 `CKPT_ROOT`、`STEPS` 和 `MODEL_PREFIX`，然后重新运行即可。

### Q: 环境变量 VLLM_WORKER_MULTIPROC_METHOD
A: vLLM v0.8.5 需要设置 `export VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免 CUDA fork 错误。脚本中已自动设置。
