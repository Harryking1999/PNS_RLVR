# PNS RLVR 代码审查总结

## 一、项目目标

在 GRPO/DAPO 强化学习训练中，引入 **真实 PNS（Process Necessity Score）** 作为辅助奖励，注入到 advantage 函数中，实现 token 级别的 PRM（Process Reward Model）。

核心思路：
1. 从训练 batch 的 rollout 中，按 entropy 选出高熵句子（step）
2. 对这些句子做 **counterfactual ablation**：构造"禁止使用该步骤"的 prompt，让模型重新生成 N 次
3. 根据消融后的正确率计算 PNS 值
4. 将 PNS bonus 加到 advantage 上

## 二、PNS 计算公式

- **正样本**（原始 rollout 答对了，acc=1）：
  - `PNS = 1 - n_correct_ablation / N`
  - `bonus = λ × (PNS - 0.5)`  → PNS<0.5 的步骤被惩罚（不重要），PNS>0.5 的被奖励（关键步骤）
- **负样本**（原始 rollout 答错了，acc=0）：
  - `PNS = -(n_correct_ablation / N)`
  - `bonus = λ × PNS`  → PNS=-1 说明干预后能答对，当前步骤很烂，狠狠惩罚；PNS=0 说明干预也没用

bonus 被加到该 step 所有 token 的 advantage 上：`advantages[b, step_start:step_end] += λ × bonus × response_mask`

## 三、新增/修改的文件

### 1. `verl/trainer/ppo/pns_utils.py`（新文件，~760 行）

核心模块，包含完整 PNS pipeline：

| 函数 | 功能 |
|------|------|
| `precompute_step_boundary_lookup(tokenizer)` | 预计算 token_id → 是否句子边界 的 bool 查找表 |
| `segment_response_into_steps(response_ids, response_mask, boundary_lookup)` | 按句子边界把 response 切成 steps |
| `select_pns_steps(entropy, response_mask, ...)` | 选高熵 token → 给 step 打分 → 全局选 top candidates |
| `build_counterfactual_prompts(candidates, input_ids, ...)` | 为每个候选 step 构建消融 prompt（multi-turn chat 格式） |
| `prepare_ablation_batch(candidates, tokenizer, max_prompt_len)` | 将消融 prompt 打包成 DataProto（left-padded） |
| `score_ablation_rollouts(ablation_output, candidates, tokenizer, compute_score_fn, num_rollouts)` | 对消融 rollout 打分，计算 PNS 值 |
| `inject_pns_into_advantage(advantages, response_mask, candidates, pns_lambda)` | 将 PNS bonus 写入 advantage tensor |
| `compute_pns_for_batch(batch, entropy, ..., config)` | **主编排函数**，串联以上所有步骤 |

### 2. `verl/trainer/ppo/ray_trainer.py`

- **`fit()` 初始化**：当 `pns_rlvr_enable=True` 时，预计算 `_pns_boundary_lookup`
- **`old_log_prob` 阶段**：保存 `batch_entropys = entropys.clone()` 供 PNS 使用
- **新增 PNS ablation 阶段**（在 `compute_advantage` 之后、`update_critic` 之前）：
  - 构建 `pns_generate_fn` wrapper，内含 **dp padding/unpadding** 逻辑
  - 调用 `compute_pns_for_batch`，传入 `self.reward_fn.compute_score`（底层打分函数）
  - 更新 `batch.batch["advantages"]`

### 3. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

- 从 `prompts.meta_info["pns_num_rollouts"]` 读取 n，覆盖 `SamplingParams.n`
- 当 `n > 1` 时，`repeat_interleave` prompt 张量 (`idx`, `attention_mask`, `position_ids`) 以匹配 `batch_size × n` 的 response

### 4. `verl/workers/config/actor.py`

新增 4 个配置参数：
- `pns_rlvr_enable: bool = False` — 是否启用 PNS
- `pns_lambda: float = 0.5` — PNS bonus 缩放因子
- `pns_num_rollouts: int = 5` — 每个消融任务生成多少条 rollout
- `pns_step_ratio: float = 0.5` — batch 中用于 PNS 测试的步骤比例（`max_steps = B × pns_step_ratio`）

### 5. `verl/trainer/ppo/core_algos.py`

- `get_step_entropy_top_mask`：移除了 Simpson 的 proxy PNS 逻辑（`return_step_pns` 参数）
- `compute_policy_loss_vanilla`：移除了 `pns_token_bonus` 参数

### 6. `verl/workers/actor/dp_actor.py`

- 移除了 `update_actor` 中 Simpson 的 proxy PNS 计算逻辑（`step_pns_token`, `pns_token_bonus`）
- 保留了 Step Forking DAPO 的 entropy_top_mask 计算

### 7. `recipe/.../run_step_forking_dapo_qwen3_1_7B.sh`

新增启动参数：`pns_rlvr_enable`, `pns_lambda`, `pns_num_rollouts`, `pns_step_ratio`

## 四、兼容性要求

代码必须兼容以下 4 种模式：
1. **纯 GRPO** — `pns_rlvr_enable=False`，不走 PNS 路径
2. **纯 DAPO** — `pns_rlvr_enable=False`，不走 PNS 路径
3. **GRPO + PNS** — `pns_rlvr_enable=True`
4. **DAPO + PNS** — `pns_rlvr_enable=True`

当 `pns_rlvr_enable=False` 时，所有 PNS 代码都不会执行（通过 `if pns_enable` 守卫）。

## 五、已修复的 Bug

### Bug 1 🔴（已修）reward_fn 签名不匹配
- **问题**：`pns_utils.py` 以 `(data_source, solution_str, ground_truth, extra_info)` 调用 reward_fn，但 `self.reward_fn` 是 `AbstractRewardManager`，其 `__call__` 签名是 `(data: DataProto)`
- **修复**：传入 `self.reward_fn.compute_score`（底层打分函数），参数名改为 `compute_score_fn`

### Bug 2 🔴（已修）response_mask KeyError
- **问题**：`score_ablation_rollouts` 访问 `ablation_output.batch["response_mask"]`，但 `generate_sequences` 返回的 TensorDict 没有这个 key
- **修复**：从 `attention_mask[:, -response_length:]` 计算 response_mask

### Bug 3 🟠（已修）ablation batch 未 pad 到 dp_size
- **问题**：`pns_generate_fn` 直接调用 `generate_sequences` 不做 padding，多卡 dispatch 会失败
- **修复**：加 `pad_dataproto_to_divisor` + `unpad_dataproto`（unpad 量 = `pad_size × pns_num_rollouts`）

### Issue 4 🟡（已修）冗余 prepare_ablation_batch
- **问题**：对全量 candidates 调用一次 `prepare_ablation_batch`，张量没用上
- **修复**：删掉冗余调用，eos/pad token_id 直接从 tokenizer 获取

## 六、需要验证的关键点

1. **`compute_score_fn` 调用**：确认 `score_ablation_rollouts` 中的调用签名与 `default_compute_score(data_source, solution_str, ground_truth, extra_info)` 一致
2. **response_mask 计算**：`attention_mask[:, -response_length:]` 是否正确提取了 response 部分的 mask
3. **n > 1 时 vLLM 输出顺序**：`vllm_rollout_spmd.py` 中 `repeat_interleave` 是否与 vLLM 输出的 prompt-interleaved 顺序匹配
4. **dp padding/unpadding**：`unpad_dataproto(output, pad_size * pns_num_rollouts)` 在 n>1 时是否正确去掉了 padded prompts 的所有 responses
5. **纯 GRPO/DAPO 模式**：`pns_rlvr_enable=False` 时，确认所有 PNS 路径都被跳过，无副作用
6. **`core_algos.py` 和 `dp_actor.py`**：确认 Simpson 的 proxy PNS 逻辑已完全移除，不影响 Step Forking DAPO 的正常 entropy mask 功能
7. **counterfactual prompt 构造**：multi-turn chat template 是否合理（user→assistant→user 三轮格式）
8. **效率**：sub-batch 大小 64、vLLM n 参数、prompt 长度过滤是否合理
