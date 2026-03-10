# PNS_RLVR 代码修改文档

> 基于 [Beyond the 80/20 Rule](https://arxiv.org/abs/2506.01939) 论文开源代码（verl 框架），本文档描述我们在训练方法上做的全部代码修改。

---

## 一、整体概览

我们的修改基于 verl 框架的 commit `6cf90ceb`，在原论文的 **Token-Level Entropy Forking DAPO** 基础上，新增了 **Step-Level Forking DAPO**（Step Forking DAPO），并配套了 PNS 验证工具和评测脚本。核心思想是：

| 方法 | 粒度 | 关键参数 | 说明 |
|------|------|---------|------|
| 原论文 Entropy Forking DAPO | Token 级 | `entropy_top_ratio=0.2` | 只在 top 20% 高熵 token 上计算梯度 |
| **新增 Step Forking DAPO** | Step（句子）级 | `entropy_top_ratio=0.2` + `step_entropy_top_ratio=0.1` | 先用高熵 token 给步骤打分，再选 top 10% 高熵步骤计算梯度 |

---

## 二、模型与数据集

### 2.1 基座模型

| 模型 | HuggingFace 地址 | 参数量 | 用途 |
|------|------------------|--------|------|
| **Qwen3-1.7B-Base** | Qwen/Qwen3-1.7B-Base | 1.7B | 小规模快速实验（4×H100 即可跑） |
| **Qwen3-14B-Base** | Qwen/Qwen3-14B-Base | 14B | 论文推荐规模，128 卡训练 |

> 论文结论：RLVR with top 20% 高熵 token 在 14B 及以上规模时效果提升显著。1.7B 主要用于快速验证和调参。

### 2.2 训练数据集

| 数据集 | 大小 | 来源 | 下载方式 |
|--------|------|------|---------|
| **math\_\_combined\_54.4k** | 54,400 条数学题

> 这是论文推荐的训练集，比原论文使用的 DAPO-Math-17k 更全面。

### 2.3 评测数据集

| 数据集 | 大小 | 说明 | 来源 |
|--------|------|------|------|
| **MATH500** | 500 题 | 数学推理 benchmark | |
| **AIME24** | 30 题 × 32 重复 = 960 条 | 美国数学邀请赛 2024，重复 32 次以计算 avg@32 | 同上，经 `duplicate_aime.sh` 扩增 |
| **AIME25** | 30 题 | 美国数学邀请赛 2025 | `tools/eval_benchmarks.py` 中内置 |
| **MATH500\_Noop** | 500 题 | MATH500 的 no-operation 变体（鲁棒性测试） | `tools/eval_benchmarks.py` 中内置 |

### 2.4 数据准备

一键脚本自动完成下载、过滤、AIME 扩增：

```bash
bash recipe/rlvr_with_high_entropy_tokens_only/prepare_train_test_datasets.sh
```

该脚本会：
1. 下载训练集 `math__combined_54.4k.parquet` → `data/`
2. 下载评测集 `math__aime_repeated_8x_240.parquet` 和 `math__math_500.parquet` → `data/`
3. 过滤评测集，只保留必要字段（`data_source`, `prompt`, `reward_model`, `extra_info`）
4. 将 AIME 扩增到 32× 重复 → `data/math__aime_repeated_32x_960.parquet`

### 2.5 PNS 分析使用的模型

PNS 工具 (`PNS_test/`) 支持对任意 HuggingFace 格式的 checkpoint 做分析。我们的实验中使用了以下模型：

| 类别 | 模型 | 说明 |
|------|------|------|
| 基线 | Qwen3-1.7B-Base | 未经 RL 训练 |
| 基线 | Qwen2.5-1.5B-Base / Instruct | 对比 SFT 效果 |
| 蒸馏 | DeepSeek-R1-Distill-Qwen-1.5B | 蒸馏模型 |
| 蒸馏+RL | DeepScaleR-1.5B-Preview | 蒸馏后再 RL |
| DAPO 训练 | DAPO-step{200..2550} | Baseline DAPO 各 checkpoint |
| Forking DAPO 训练 | Forking-DAPO-step{200..2550} | 我们的方法各 checkpoint |

> 使用 `PNS_test/launch_dynamic_pns.sh` 可以一次性在多 GPU 上并行跑所有 checkpoint 的 PNS 分析。

---

## 三、核心代码修改

### 3.1 新增配置参数

**文件**: `verl/workers/config/actor.py`

```python
# 原有参数（论文方法）
entropy_top_ratio: Optional[float] = None  # top 20% 高熵 token

# 新增参数（Step Forking）
step_entropy_top_ratio: Optional[float] = None  # top 10% 高熵步骤
```

**文件**: `verl/trainer/config/actor/actor.yaml`

```yaml
entropy_top_ratio: null
step_entropy_top_ratio: null
```

### 3.2 Step 切分与步骤级 Entropy Mask

**文件**: `verl/trainer/ppo/core_algos.py`

新增两个函数：

#### `get_global_entropy_top_mask(entropy, response_mask, top_ratio)`
- 原论文方法的实现：在整个 batch 的 response token 中，选择 top `top_ratio` 高熵 token
- 返回 `[B, S]` 的 binary mask

#### `get_step_entropy_top_mask(entropy, response_mask, response_ids, boundary_lookup, token_top_ratio, step_top_ratio)`
- **Step Forking DAPO 的核心算法**
- 流程：
  1. 对 batch 中的每个样本，使用预计算的 `boundary_lookup`（换行符 `\n` + 句末标点 `.?!。？！`）将 response 切分为步骤
  2. 为每个步骤打分：步骤内 top `token_top_ratio` 高熵 token 的熵值之和
  3. 全局选择 top `step_top_ratio` 比例的步骤
  4. 返回 `[B, S]` 的 binary mask，被选中步骤内的**所有 token** 都为 1

### 3.3 Actor 训练时的 Mask 应用

**文件**: `verl/workers/actor/dp_actor.py`

#### 初始化阶段：预计算步骤边界
- 当 `step_entropy_top_ratio > 0` 时，在 `__init__` 中调用 `_precompute_step_boundary_lookup(tokenizer)`
- 遍历 tokenizer 的全部 vocab，构建一个 `[vocab_size]` 的 boolean lookup table：如果 token 解码后包含 `\n` 或以 `.?!。？！` 结尾，则标记为边界 token
- 这个预计算只在初始化时执行一次，避免训练中的重复开销

#### 训练阶段：动态选择 Mask 模式
```python
# update_actor() 中的关键逻辑：
if step_entropy_top_ratio is not None and step_boundary_lookup is not None:
    # Step Forking DAPO: 步骤级 mask
    entropy_top_mask = get_step_entropy_top_mask(...)
else:
    # 原论文: Token 级 mask
    entropy_top_mask = get_global_entropy_top_mask(...)
```

#### Loss 计算时的 Mask 使用
在 `compute_policy_loss_vanilla()` 中：
```python
if entropy_top_mask is None:
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, ...)
else:
    # 用 entropy_top_mask 进一步过滤，只在选中的 token/步骤上计算 loss
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask * entropy_top_mask, ...)
```

---

## 四、训练脚本

### 4.1 Token-Level Forking DAPO（原论文方法）

**脚本**: `recipe/rlvr_with_high_entropy_tokens_only/run_dapo_qwen3_1_7B.sh`

- 模型: Qwen3-1.7B-Base，4×H100
- 关键参数: `entropy_top_ratio=0.2`（只在 top 20% 高熵 token 上算梯度）

### 4.2 Step Forking DAPO（新增方法）

**脚本**: `recipe/rlvr_with_high_entropy_tokens_only/run_step_forking_dapo_qwen3_1_7B.sh`

- 模型: Qwen3-1.7B-Base，4×H100
- 关键参数:
  - `entropy_top_ratio=0.2`：步骤内 top 20% 高熵 token 用于步骤打分
  - `step_entropy_top_ratio=0.1`：全局选 top 10% 高熵步骤做梯度下降

### 4.3 对照实验

| 脚本 | 方法 | entropy_top_ratio | step_entropy_top_ratio |
|------|------|-------------------|------------------------|
| `run_dapo_qwen3_14b.sh` | Baseline DAPO | 无 | 无 |
| `run_only_top20_..._14b.sh` | Token Forking | 0.2 | 无 |
| `run_dapo_qwen3_1_7B.sh` | Token Forking (1.7B) | 0.2 | 无 |
| `run_step_forking_..._1_7B.sh` | **Step Forking (1.7B)** | 0.2 | 0.1 |

---

## 五、评测与分析工具

### 5.1 Benchmark 评测 (`tools/eval_benchmarks.py`)

- 支持 AIME24、AIME25、MATH500、MATH500_Noop 四个 benchmark
- 使用 vLLM 进行批量推理，支持 Greedy (temp=0, n=1) 和 Sampling (temp=1.0, n=16) 两种模式
- 指标：Acc@1, Len@1, Acc@16, Pass@16, Len@16

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
    --models /path/to/merged_model \
    --benchmarks aime24 aime25 math500 \
    --max_tokens 16384
```

### 5.2 Step Entropy 可视化 (`tools/visualize_step_entropy.py`)

- 用于验证「高熵 token」和「高熵步骤」之间是否有人类可理解的一致关系
- 支持两种步骤切分方法：
  - `punctuation`：按换行符 + 句末标点切分（默认）
  - `double_newline`：按 `\n\n` 切分
- 支持缓存机制（`--save_cache` / `--load_cache`），从缓存加载时不需要 GPU

### 5.3 PNS (Process Necessity Score) 验证 (`PNS_test/compute_pns.py`)

- 通过消融实验估计每个推理步骤的必要性
- PNS 定义：对正确解答中的每个步骤，移除该步骤后让模型 rollout N 次，`PNS = 1 - (正确数/N)`
- 支持两种消融模式：
  - `truncate`：截断后续所有步骤，重新生成
  - `counterfactual`：仅删除目标步骤，保留后续上下文
- 用于验证 Step Forking DAPO 选中的高熵步骤是否确实是推理中的关键步骤

---

## 六、文件变更总览

```
verl/workers/config/actor.py           # 新增 step_entropy_top_ratio 配置
verl/trainer/config/actor/actor.yaml   # 对应 YAML 配置
verl/trainer/ppo/core_algos.py         # 新增 get_global_entropy_top_mask, get_step_entropy_top_mask
verl/workers/actor/dp_actor.py         # 预计算 boundary_lookup + 训练时 mask 选择逻辑
verl/utils/reward_score/naive_dapo.py  # 新增数学评分器（支持 AIME/MATH 等）

recipe/rlvr_with_high_entropy_tokens_only/
├── run_dapo_qwen3_1_7B.sh                 # Token Forking DAPO (1.7B)
├── run_step_forking_dapo_qwen3_1_7B.sh    # Step Forking DAPO (1.7B)
├── run_dapo_qwen3_14b.sh                  # Baseline DAPO (14B)
└── run_only_top20_..._14b.sh              # Token Forking DAPO (14B)

tools/
├── eval_benchmarks.py          # Benchmark 评测脚本
└── visualize_step_entropy.py   # Step Entropy 可视化工具

PNS_test/
├── compute_pns.py              # PNS 计算主脚本
├── compare_pns_results.py      # PNS 结果对比分析
├── inspect_pns_rollouts.py     # PNS rollout 检查
├── launch_dynamic_pns.sh       # 多 GPU 并行 PNS 计算
└── run_pns_6models.sh          # 6 模型批量 PNS 估算
```

---

## 七、可复用模块与 PNS_RLVR 开发指南

> 本节面向合作者，梳理现有代码中哪些模块可以直接复用于 PNS_RLVR 算法的开发，以及建议的扩展方式。

### 7.1 可复用模块一览

| # | 模块 | 位置 | 可复用的场景 |
|---|------|------|-------------|
| 1 | **步骤切分** | `PNS_test/compute_pns.py::segment_into_steps()` | 将 response 切分为推理步骤 |
| 2 | **GPU 高效步骤切分（boundary lookup）** | `verl/workers/actor/dp_actor.py::_precompute_step_boundary_lookup()` | 训练时 O(1) 判断 token 是否是步骤边界 |
| 3 | **Token/Step 级 Mask 生成** | `verl/trainer/ppo/core_algos.py::get_global_entropy_top_mask()` / `get_step_entropy_top_mask()` | 选择哪些 token/步骤参与梯度计算 |
| 4 | **Mask → Policy Loss 接口** | `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla()` 的 `entropy_top_mask` 参数 | 任何 token 选择策略都可以通过这个 mask 接口注入 |
| 5 | **步骤打分（Entropy-based）** | `PNS_test/compute_pns.py::compute_step_entropy_scores()` + `select_top_steps()` | 基于 entropy 的步骤重要性排序 |
| 6 | **PNS 消融 rollout 流水线** | `PNS_test/compute_pns.py` Phase 3 | 删除步骤后让模型重新 rollout，验证步骤必要性 |
| 7 | **数学答案验证器** | `verl/utils/reward_score/naive_dapo.py::compute_score()` | 判断模型输出是否正确（支持 LaTeX、分数等复杂格式） |
| 8 | **Benchmark 评测** | `tools/eval_benchmarks.py` | 评估模型在 AIME/MATH500 上的表现 |
| 9 | **配置系统扩展模式** | `verl/workers/config/actor.py` + `verl/trainer/config/actor/actor.yaml` | 新增超参的标准做法 |
| 10 | **多 GPU 并行调度** | `PNS_test/launch_dynamic_pns.sh` | 多 checkpoint 并行评测的 GPU 队列调度 |

### 7.2 详细说明

#### ① 步骤切分（直接复用）

两套实现，功能等价，适用场景不同：

**离线分析用**（Python list 操作，灵活、可 debug）：
```python
# PNS_test/compute_pns.py
steps = segment_into_steps(response_tokens)
# 返回: [{'start': 0, 'end': 15, 'text': '...', 'token_indices': [0,1,...14]}, ...]
```

**训练时 GPU 用**（预计算 lookup table，O(1) 查表）：
```python
# verl/workers/actor/dp_actor.py
# 初始化时：遍历 vocab 构建 boundary_lookup[vocab_size] (bool tensor)
# 训练时：boundary_lookup[token_id] 即可判断是否为步骤边界
```

切分规则：遇到包含 `\n` 或以 `.?!。？！` 结尾的 token 即为边界，边界 token 属于当前步骤的最后一个 token。

> **如果 PNS_RLVR 需要更换切分策略**（比如用 `\n\n` 切分、按语义切分等），只需修改 `segment_into_steps()` 和 `_precompute_step_boundary_lookup()` 这两处即可，下游的打分和 mask 逻辑完全不需要改。

#### ② Mask 机制（核心复用点）

现有系统已经建好了从「步骤选择」到「Loss 计算」的完整管线：

```
步骤选择策略 → 生成 entropy_top_mask [B, S] → 传入 policy_loss_fn → mask 掉不参与的 token
```

**关键接口**在 `compute_policy_loss_vanilla()` (core_algos.py L1105-1108)：
```python
# entropy_top_mask 是一个 [B, S] 的 0/1 tensor
# loss_mask = response_mask * entropy_top_mask
# 只在 mask=1 的位置计算 policy gradient loss
```

**对 PNS_RLVR 的意义**：你只需要实现一个新的 mask 生成函数（比如 `get_pns_based_mask()`），生成同样形状的 `[B, S]` mask，然后通过同一个 `entropy_top_mask` 接口传入即可。不需要改 policy loss 的计算逻辑。

#### ③ PNS 消融 rollout（直接复用）

`PNS_test/compute_pns.py` 实现了三种消融模式：

| 模式 | 做法 | 适用场景 |
|------|------|---------|
| `counterfactual` | 保留前文 A，告诉模型"不要用步骤 B" | 原始 PNS 论文方法，最接近因果推断 |
| `truncate` | 截断到步骤前，从头重新生成 | 测试"没有这步及之后的内容，能否做对" |
| `skip` | 跳过目标步骤但保留后续步骤 | 测试"缺少这步但给后续 context，能否做对" |

每种模式的 prefix 构建函数（`build_ablation_prefix_*`）可以直接复用。如果 PNS_RLVR 需要在训练时做 online PNS 估算，可以将 Phase 3 的逻辑集成到 rollout worker 中。

#### ④ 步骤打分（可替换）

当前的步骤打分用的是 entropy-based 方法：
```python
# PNS_test/compute_pns.py
step_scores = compute_step_entropy_scores(steps, token_entropies, token_top_ratio=0.2)
selected = select_top_steps(steps, step_scores, step_top_ratio=0.1)
```

**PNS_RLVR 可以替换为 PNS-based 打分**，只需实现一个新的 `compute_pns_step_scores()` 函数即可，后续的 `select_top_steps()` 和 mask 生成逻辑不变。

#### ⑤ 数学答案验证器（直接复用）

```python
from verl.utils.reward_score.naive_dapo import compute_score
result = compute_score(model_output_text, ground_truth_answer)
# result["acc"] = 1.0 or 0.0
```

支持 LaTeX `\boxed{}`、分数、科学计数法等复杂数学表达式的匹配。PNS 消融 rollout 和 benchmark 评测都依赖它。

### 7.3 PNS_RLVR 算法集成建议

当前 PNS 是一个**离线分析工具**（先跑完训练，再用 PNS 事后分析）。如果要把 PNS 集成到训练中，有以下几种可行路径：

#### 路径 A：PNS 作为步骤级 Reward（改 reward）

在 `recipe/dapo/dapo_ray_trainer.py` 的 reward 计算阶段，对每个 response：
1. 切分为步骤（复用 `segment_into_steps`）
2. 对每个步骤做消融 rollout（复用 Phase 3 的 `build_ablation_prefix_*`）
3. 计算 PNS 作为 step-level reward

**优点**：与现有 advantage estimation 无缝对接
**挑战**：消融 rollout 开销大，可能需要异步/缓存机制

#### 路径 B：PNS 作为步骤级 Mask（改 mask，复用现有管线最多）

类似 Step Forking DAPO 的做法，但用 PNS 替代 entropy 打分：
1. 切分为步骤（复用 boundary lookup）
2. 用 PNS 估算每步重要性
3. 生成 `pns_mask [B, S]`（复用 mask → loss 接口）

**核心修改点**：
- `dp_actor.py` 的 `update_actor()` 中，新增 `if pns_mode: pns_mask = get_pns_mask(...)`
- `core_algos.py` 中新增 `get_pns_mask()` 函数
- 通过已有的 `entropy_top_mask` 接口传入 policy loss

#### 路径 C：混合模式（Entropy 初筛 + PNS 精筛）

先用 entropy（零开销）粗筛候选步骤，再对候选步骤做 PNS 验证。这样可以大幅降低 rollout 开销。

### 7.4 新增超参的标准流程

如果需要添加新的配置参数（如 `pns_top_ratio`），需要同时修改以下三处：

1. **Python dataclass**: `verl/workers/config/actor.py` 的 `ActorConfig` 类中添加字段
2. **YAML 默认值**: `verl/trainer/config/actor/actor.yaml` 中添加默认值
3. **Generated YAML**: `verl/trainer/config/_generated_ppo_trainer.yaml` 和 `_generated_ppo_megatron_trainer.yaml`

然后在 `dp_actor.py` 的 `update_actor()` 中通过 `self.config.get('pns_top_ratio', None)` 读取。

---

## 八、快速上手

```bash
# 1. 数据准备
bash recipe/rlvr_with_high_entropy_tokens_only/prepare_train_test_datasets.sh

# 2. 运行 Step Forking DAPO 训练 (4×H100, Qwen3-1.7B-Base)
bash recipe/rlvr_with_high_entropy_tokens_only/run_step_forking_dapo_qwen3_1_7B.sh

# 3. 评测
CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
    --models /path/to/checkpoint --benchmarks aime24 math500

# 4. PNS 验证
CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
    --model /path/to/checkpoint --num_problems 30 --num_rollouts 5
```
