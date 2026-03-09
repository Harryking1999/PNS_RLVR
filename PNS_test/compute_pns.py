#!/usr/bin/env python3
"""
PNS (Process Necessity Score) Estimation Tool

Purpose:
  Estimate the necessity of each reasoning step by ablation, to evaluate
  whether PNS-based rewards could improve RLVR training.

PNS Definition:
  For each step in a correct solution:
    1. Remove the step (via truncation or skip)
    2. Let the model rollout N times from the ablated context
    3. PNS = 1 - (num_correct_rollouts / N)
  Higher PNS → the step is more necessary for reaching the correct answer.

Ablation Modes:
  - truncate: Prefix = prompt + steps_before_i → regenerate everything
              Tests: "Is this a pivot step? Can the model solve it without
                      this step and everything after?"
  - skip:     Prefix = prompt + steps_before_i + steps_after_i (remove step i
              from the middle, but keep subsequent steps minus the answer step)
              → regenerate the final answer
              Tests: "Is this step necessary given all other reasoning?"

Workflow (3 phases, each saves checkpoints for resumability):
  Phase 1: Generate correct responses with vLLM
  Phase 2: Compute per-token entropy + step selection with HF model
  Phase 3: Ablation rollouts with vLLM → compute PNS

Usage:
    # Basic run
    CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
        --model /ssdwork/fuzhizhang/merged_models/Step-Forking-DAPO-step1600 \
        --num_problems 50 --num_rollouts 5

    # Test all steps (not just high-entropy), for correlation analysis
    CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
        --model /ssdwork/fuzhizhang/merged_models/Step-Forking-DAPO-step1600 \
        --test_all_steps --num_problems 30 --num_rollouts 5

    # Resume from a previous interrupted run
    CUDA_VISIBLE_DEVICES=0 python PNS_test/compute_pns.py \
        --model /ssdwork/fuzhizhang/merged_models/Step-Forking-DAPO-step1600 \
        --resume
"""

import argparse
import gc
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

# CRITICAL: Must set spawn method BEFORE importing torch/CUDA
# to avoid "Cannot re-initialize CUDA in forked subprocess" in vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from verl.utils.reward_score.naive_dapo import compute_score


# ──────────────────────────── Constants ────────────────────────────

SUFFIX = "\nPlease output the final answer within \\boxed{}."
SENTENCE_ENDINGS = {'.', '?', '!', '。', '？', '！'}


# ──────────────────────────── Data Loading ────────────────────────

def load_math500():
    """Load MATH500 benchmark."""
    path = "/home/fuzhizhang.fzz/data/math__math_500.parquet"
    df = pd.read_parquet(path)
    problems = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            content = prompt[0]["content"] if isinstance(prompt[0], dict) else str(prompt[0])
        else:
            content = str(prompt)
        gt = row["reward_model"]["ground_truth"]
        problems.append({
            "id": f"math500_{idx}",
            "problem": content,
            "answer": str(gt),
        })
    return problems


def load_aime24():
    """Load AIME 2024 benchmark."""
    path = "/home/fuzhizhang.fzz/data/aime_2024/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    problems = []
    for _, row in df.iterrows():
        problems.append({
            "id": f"aime24_{row['id']}",
            "problem": row["problem"],
            "answer": str(row["answer"]),
        })
    return problems


BENCHMARK_LOADERS = {
    "math500": load_math500,
    "aime24": load_aime24,
}


# ──────────────────────────── Prompt Building ─────────────────────

def build_prompt(problem_text: str, tokenizer) -> str:
    """Build chat-template prompt for a problem."""
    if "\\boxed{}" in problem_text and "Please output" in problem_text:
        user_content = problem_text
    else:
        user_content = problem_text + SUFFIX
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ──────────────────────────── Step Segmentation ───────────────────

def segment_into_steps(response_tokens):
    """
    Segment response tokens into reasoning steps.

    Splitting rule: a step boundary occurs at tokens containing '\\n' or
    ending with sentence-ending punctuation (. ? ! 。 ？ ！).
    The boundary token belongs to the current step.

    Returns:
        List of step dicts with keys: start, end, token_indices, text
    """
    steps = []
    current_start = 0

    for i, tok_text in enumerate(response_tokens):
        stripped = tok_text.rstrip()
        is_boundary = False

        if '\n' in tok_text:
            is_boundary = True
        if stripped and stripped[-1] in SENTENCE_ENDINGS:
            is_boundary = True

        if is_boundary and i < len(response_tokens) - 1:
            steps.append({
                'start': current_start,
                'end': i + 1,
                'token_indices': list(range(current_start, i + 1)),
                'text': ''.join(response_tokens[current_start:i + 1]),
            })
            current_start = i + 1

    # Last segment
    if current_start < len(response_tokens):
        steps.append({
            'start': current_start,
            'end': len(response_tokens),
            'token_indices': list(range(current_start, len(response_tokens))),
            'text': ''.join(response_tokens[current_start:]),
        })

    return steps


# ──────────────────────────── Entropy Computation ─────────────────

def compute_token_entropy_batch(model, tokenizer, items, device="cuda:0",
                                max_seq_len=32768):
    """
    Compute per-token entropy for a list of (prompt, response_token_ids) items.
    Processes one at a time to handle variable-length sequences.

    Args:
        model: HuggingFace model in eval mode
        tokenizer: corresponding tokenizer
        items: list of dicts with 'prompt' and 'response_token_ids'
        device: CUDA device
        max_seq_len: max sequence length to process (skip longer)

    Returns:
        List of np.ndarray, each of shape (response_len,)
    """
    all_entropies = []

    for item in items:
        prompt_text = item["prompt"]
        response_ids = item["response_token_ids"]

        input_ids = tokenizer(str(prompt_text), return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]

        # Build full sequence: prompt + response
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(response_ids, dtype=torch.long)
        ]).unsqueeze(0).to(device)

        total_len = full_ids.shape[1]

        if total_len > max_seq_len:
            print(f"    [WARNING] Sequence too long ({total_len} > {max_seq_len}), "
                  f"using zero entropy fallback")
            all_entropies.append(np.zeros(len(response_ids)))
            continue

        with torch.no_grad():
            output = model(full_ids)
            logits = output.logits[0]  # (total_len, vocab_size)

        # logits[t] predicts token at position t+1
        response_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids)]

        # Entropy = -sum(p * log p)
        probs = torch.softmax(response_logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        token_entropies = -(probs * log_probs).sum(dim=-1).cpu().numpy()

        all_entropies.append(token_entropies)

        # Free memory
        del logits, response_logits, probs, log_probs, output
        torch.cuda.empty_cache()

    return all_entropies


def compute_step_entropy_scores(steps, token_entropies, token_top_ratio=0.2):
    """
    Score each step by cumulative entropy of its top high-entropy tokens.

    Method:
    1. Find global top token_top_ratio high-entropy tokens
    2. For each step, sum the entropies of high-entropy tokens within it

    Returns:
        step_scores: list of floats
        high_entropy_mask: np.ndarray of bools
    """
    entropies = np.array(token_entropies)
    n_tokens = len(entropies)
    if n_tokens == 0:
        return [], np.array([], dtype=bool)

    top_k = max(1, int(np.ceil(n_tokens * token_top_ratio)))
    top_indices = np.argsort(entropies)[-top_k:]
    high_entropy_mask = np.zeros(n_tokens, dtype=bool)
    high_entropy_mask[top_indices] = True

    step_scores = []
    for step in steps:
        indices = np.array(step['token_indices'])
        mask_in_step = high_entropy_mask[indices]
        entropy_in_step = entropies[indices]
        score = float(entropy_in_step[mask_in_step].sum())
        step_scores.append(score)

    return step_scores, high_entropy_mask


def select_top_steps(steps, step_scores, step_top_ratio=0.2):
    """Select the top step_top_ratio fraction of high-entropy steps."""
    n_steps = len(steps)
    if n_steps == 0:
        return set()
    top_k = max(1, int(np.ceil(n_steps * step_top_ratio)))
    sorted_indices = np.argsort(step_scores)[-top_k:]
    return set(sorted_indices.tolist())


# ──────────────────────────── Ablation Prefix Construction ────────

# Template for mid-solution counterfactual (step_idx > 0, has prefix A)
COUNTERFACTUAL_MID_TEMPLATE = (
    "You have been working on this problem and have made progress as shown above. "
    "However, for the next step, you must NOT use the following reasoning:\n"
    "\"{FORBIDDEN}\"\n\n"
    "Please find an alternative approach to continue solving the problem. "
    "Do NOT repeat or rephrase the forbidden step. "
    "Output the final answer within \\boxed{}."
)

# Template for first-step counterfactual (step_idx == 0, no prefix)
COUNTERFACTUAL_FIRST_STEP_TEMPLATE = (
    "When solving this problem, do NOT begin with or use the following reasoning approach:\n"
    "\"{FORBIDDEN}\"\n\n"
    "Find a different way to start solving the problem. "
    "Do NOT repeat or rephrase the forbidden approach. "
    "Output the final answer within \\boxed{}."
)


def build_ablation_prefix_counterfactual(problem_text, response_tokens, step, tokenizer):
    """
    Counterfactual mode (original PNS paper method):

    For step B with prefix A:
      - Keep A (everything the model generated before B)
      - Tell the model: "don't use B, find an alternative B'"
      - Model generates B' and continues solving

    When step_idx=0 (first step, prefix A is empty):
      - Single user turn: problem + "don't start with B"
      - No empty assistant turn (which would confuse the model)

    When step_idx>0 (has prefix A):
      - User: original problem
      - Assistant: prefix A (progress so far)
      - User: "don't use B, find alternative"
    """
    # text before the target step (the "progress so far")
    prefix_text = ''.join(response_tokens[:step['start']])

    # the forbidden step text
    forbidden_step_text = step['text'].strip()

    # Build user content (the original problem)
    if "\\boxed{}" in problem_text and "Please output" in problem_text:
        user_content = problem_text
    else:
        user_content = problem_text + SUFFIX

    if step['start'] == 0:
        # First step: no prefix, single user turn
        counterfactual_msg = COUNTERFACTUAL_FIRST_STEP_TEMPLATE.replace(
            "{FORBIDDEN}", forbidden_step_text
        )
        messages = [
            {"role": "user", "content": user_content + "\n\n" + counterfactual_msg},
        ]
    else:
        # Mid-solution: multi-turn with prefix
        counterfactual_msg = COUNTERFACTUAL_MID_TEMPLATE.replace(
            "{FORBIDDEN}", forbidden_step_text
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prefix_text},
            {"role": "user", "content": counterfactual_msg},
        ]

    ablation_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return ablation_prompt


def build_ablation_prefix_truncate(prompt, response_tokens, step, tokenizer=None):
    """
    Truncate mode: prefix = prompt + all tokens BEFORE the target step.
    The model regenerates from this point.
    """
    prefix_text = ''.join(response_tokens[:step['start']])
    return prompt + prefix_text


def build_ablation_prefix_skip(prompt, response_tokens, steps, target_step_idx, tokenizer=None):
    """
    Skip mode: prefix = prompt + all steps except the target step
    (and excluding the answer step if it's the last one).
    """
    # Find the answer step (last step containing \\boxed{)
    answer_step_idx = None
    for j in range(len(steps) - 1, -1, -1):
        if '\\boxed{' in steps[j]['text'] or '\\boxed ' in steps[j]['text']:
            answer_step_idx = j
            break

    # Build prefix by concatenating all steps except target and answer
    parts = []
    for j, step in enumerate(steps):
        if j == target_step_idx:
            continue  # skip the target step
        if j == answer_step_idx:
            continue  # skip the answer step (model should regenerate it)
        parts.append(step['text'])

    return prompt + ''.join(parts)


# ──────────────────────────── Phase Functions ─────────────────────

def phase1_generate_correct(args, phase1_path):
    """Phase 1: Use vLLM to generate responses, find correct ones."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Phase 1: Generating correct responses")
    print(f"  Model: {args.model}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load benchmark
    all_problems = BENCHMARK_LOADERS[args.benchmark]()
    if args.num_problems > 0 and args.num_problems < len(all_problems):
        np.random.seed(args.seed)
        indices = np.random.choice(len(all_problems), args.num_problems, replace=False)
        problems = [all_problems[i] for i in sorted(indices)]
    else:
        problems = all_problems

    print(f"  {len(problems)} problems to solve")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_tokens + 2048,
        dtype="bfloat16",
    )

    # Generate N samples per problem to find correct ones
    prompts = [build_prompt(p["problem"], tokenizer) for p in problems]
    sp = SamplingParams(
        temperature=1.0, max_tokens=args.max_tokens,
        n=args.num_initial_samples,
    )

    print(f"  Generating {args.num_initial_samples} samples per problem...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s")

    # Find first correct response for each problem
    correct_data = []
    for i, (prob, output) in enumerate(zip(problems, outputs)):
        for out in output.outputs:
            result = compute_score(out.text, prob["answer"])
            if bool(result["acc"]):
                correct_data.append({
                    "id": prob["id"],
                    "problem": prob["problem"],
                    "answer": prob["answer"],
                    "prompt": prompts[i],
                    "response_text": out.text,
                    "response_token_ids": list(out.token_ids),
                })
                break

    print(f"  Correct responses: {len(correct_data)}/{len(problems)} "
          f"({len(correct_data)/len(problems)*100:.1f}%)")

    with open(phase1_path, "w") as f:
        json.dump(correct_data, f, ensure_ascii=False)
    print(f"  Saved to {phase1_path}")

    del llm, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return correct_data


def phase2_entropy_steps(args, correct_data, phase2_path):
    """Phase 2: Compute per-token entropy, segment steps, select high-entropy steps."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"Phase 2: Computing entropy and step selection")
    print(f"  entropy_top_ratio: {args.entropy_top_ratio}")
    print(f"  step_entropy_top_ratio: {args.step_entropy_top_ratio}")
    print(f"{'='*60}")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    print(f"  Model loaded on {device}")

    step_data = []
    for i, item in enumerate(correct_data):
        print(f"  [{i+1}/{len(correct_data)}] Processing {item['id']}...", end="", flush=True)

        response_token_ids = item["response_token_ids"]
        response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

        # Compute entropy
        entropies_list = compute_token_entropy_batch(
            model, tokenizer, [item], device=device
        )
        token_entropies = entropies_list[0]

        # Segment into steps
        steps = segment_into_steps(response_tokens)

        # Score steps
        step_scores, high_entropy_mask = compute_step_entropy_scores(
            steps, token_entropies, token_top_ratio=args.entropy_top_ratio
        )

        # Select high-entropy steps
        selected_indices = select_top_steps(
            steps, step_scores, step_top_ratio=args.step_entropy_top_ratio
        )

        # Build step info
        step_info = []
        for j, (step, score) in enumerate(zip(steps, step_scores)):
            has_answer = bool(re.search(r'\\boxed\s*\{', step['text']))
            step_info.append({
                "step_idx": j,
                "text": step["text"],
                "start": step["start"],
                "end": step["end"],
                "entropy_score": score,
                "is_high_entropy": j in selected_indices,
                "has_answer": has_answer,
                "n_tokens": len(step["token_indices"]),
                "n_high_entropy_tokens": int(sum(
                    high_entropy_mask[idx] for idx in step["token_indices"]
                )),
            })

        # Determine position category for each step
        n_steps = len(steps)
        for j, si in enumerate(step_info):
            if n_steps <= 3:
                si["position"] = "all"
            elif j < n_steps / 3:
                si["position"] = "early"
            elif j < 2 * n_steps / 3:
                si["position"] = "middle"
            else:
                si["position"] = "late"

        step_data.append({
            "id": item["id"],
            "problem": item["problem"],
            "answer": item["answer"],
            "prompt": item["prompt"],
            "response_text": item["response_text"],
            "response_token_ids": response_token_ids,
            "response_tokens": response_tokens,
            "token_entropies": token_entropies.tolist(),
            "steps": step_info,
            "num_steps": n_steps,
            "num_high_entropy_steps": len(selected_indices),
            "mean_entropy": float(token_entropies.mean()),
        })

        print(f" {n_steps} steps, {len(selected_indices)} HE, "
              f"avg_ent={token_entropies.mean():.3f}")

    with open(phase2_path, "w") as f:
        json.dump(step_data, f, ensure_ascii=False)
    print(f"  Saved to {phase2_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return step_data


def phase3_ablation_rollouts(args, step_data, phase3_path):
    """Phase 3: Ablation rollouts with vLLM → compute PNS for each step."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Phase 3: Ablation rollouts ({args.ablation_mode} mode)")
    print(f"  num_rollouts: {args.num_rollouts}")
    print(f"  test_all_steps: {args.test_all_steps}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Collect all ablation tasks
    ablation_tasks = []
    for item_idx, item in enumerate(step_data):
        response_tokens = item["response_tokens"]
        prompt = item["prompt"]
        steps = item["steps"]

        for step in steps:
            # Skip non-high-entropy steps unless testing all
            if not args.test_all_steps and not step["is_high_entropy"]:
                continue
            # Skip the answer step itself (testing it isn't meaningful)
            if step["has_answer"]:
                continue
            # Skip very first tiny steps (< 3 tokens)
            if step["n_tokens"] < 3:
                continue

            step_idx = step["step_idx"]

            if args.ablation_mode == "counterfactual":
                ablation_prefix = build_ablation_prefix_counterfactual(
                    item["problem"], response_tokens, step, tokenizer
                )
            elif args.ablation_mode == "truncate":
                ablation_prefix = build_ablation_prefix_truncate(
                    prompt, response_tokens, step
                )
            else:  # skip
                ablation_prefix = build_ablation_prefix_skip(
                    prompt, response_tokens, steps, step_idx
                )

            ablation_tasks.append({
                "item_idx": item_idx,
                "step_idx": step_idx,
                "problem_id": item["id"],
                "answer": item["answer"],
                "ablation_prefix": ablation_prefix,
                "step_text": step["text"],
                "entropy_score": step["entropy_score"],
                "is_high_entropy": step["is_high_entropy"],
                "position": step["position"],
                "n_tokens": step["n_tokens"],
            })

    print(f"  {len(ablation_tasks)} ablation tasks × {args.num_rollouts} rollouts each")

    if len(ablation_tasks) == 0:
        print("  No ablation tasks! Check your data.")
        return []

    # ── Tokenize all prefixes and compute length statistics ──
    prefixes = [t["ablation_prefix"] for t in ablation_tasks]
    prefix_token_lens = []
    for i, pf in enumerate(prefixes):
        toks = tokenizer.encode(pf, add_special_tokens=False)
        prefix_token_lens.append(len(toks))
        ablation_tasks[i]["prefix_token_len"] = len(toks)

    max_prefix_len = max(prefix_token_lens)
    mean_prefix_len = sum(prefix_token_lens) / len(prefix_token_lens)
    median_prefix_len = sorted(prefix_token_lens)[len(prefix_token_lens) // 2]
    p95_prefix_len = sorted(prefix_token_lens)[int(len(prefix_token_lens) * 0.95)]

    print(f"  Prefix token lengths:")
    print(f"    min={min(prefix_token_lens)}, mean={mean_prefix_len:.0f}, "
          f"median={median_prefix_len}, p95={p95_prefix_len}, max={max_prefix_len}")

    # ── Determine max_model_len: cap to fit in single GPU (80GB) ──
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_max_pos = getattr(model_config, 'max_position_embeddings', 32768)
    # Use same max_model_len as Phase 1 to avoid OOM on single GPU
    max_model_len = min(args.max_tokens + 2048, model_max_pos)
    # Max prefix length that still allows meaningful generation (at least 2048 tokens)
    max_allowed_prefix = max_model_len - 2048

    # Filter out tasks whose prefix is too long
    n_before = len(ablation_tasks)
    n_skipped_long = 0
    filtered_tasks = []
    for task in ablation_tasks:
        if task["prefix_token_len"] > max_allowed_prefix:
            n_skipped_long += 1
        else:
            filtered_tasks.append(task)
    ablation_tasks = filtered_tasks
    prefixes = [t["ablation_prefix"] for t in ablation_tasks]
    prefix_token_lens = [t["prefix_token_len"] for t in ablation_tasks]

    print(f"  Model max_position_embeddings: {model_max_pos}")
    print(f"  max_model_len (capped for GPU): {max_model_len}")
    print(f"  max_allowed_prefix: {max_allowed_prefix}")
    if n_skipped_long > 0:
        print(f"  ⚠ Skipped {n_skipped_long}/{n_before} tasks with prefix > {max_allowed_prefix} tokens")
    print(f"  Remaining tasks: {len(ablation_tasks)}")

    if len(ablation_tasks) == 0:
        print("  No ablation tasks left after filtering! Check your data.")
        return []

    # Recompute prefix stats after filtering
    print(f"  Prefix token lengths (after filter):")
    print(f"    min={min(prefix_token_lens)}, mean={sum(prefix_token_lens)/len(prefix_token_lens):.0f}, "
          f"median={sorted(prefix_token_lens)[len(prefix_token_lens)//2]}, "
          f"p95={sorted(prefix_token_lens)[int(len(prefix_token_lens)*0.95)]}, "
          f"max={max(prefix_token_lens)}")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        dtype="bfloat16",
    )

    sp = SamplingParams(
        temperature=1.0, max_tokens=args.max_tokens,
        n=args.num_rollouts,
    )

    # Batch generation
    BATCH_SIZE = 32
    all_outputs = []

    for batch_start in range(0, len(prefixes), BATCH_SIZE):
        batch = prefixes[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(prefixes) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} "
              f"({len(batch)} tasks)...", end="", flush=True)
        t0 = time.time()
        outputs = llm.generate(batch, sp)
        all_outputs.extend(outputs)
        print(f" done in {time.time()-t0:.1f}s")

    # Score ablation rollouts
    ablation_results = []
    n_hit_length_limit = 0
    for task, output in zip(ablation_tasks, all_outputs):
        rollout_results = []
        for out in output.outputs:
            result = compute_score(out.text, task["answer"])
            gen_len = len(out.token_ids)
            rollout_results.append({
                "correct": bool(result["acc"]),
                "response_len": gen_len,
            })

        n_correct = sum(1 for r in rollout_results if r["correct"])
        pns = 1.0 - n_correct / args.num_rollouts

        prefix_tl = task["prefix_token_len"]
        avg_gen_len = sum(r["response_len"] for r in rollout_results) / len(rollout_results)
        max_gen_len = max(r["response_len"] for r in rollout_results)
        # Check if any rollout hit the max_tokens limit (potential truncation)
        hit_limit = any(r["response_len"] >= args.max_tokens - 10 for r in rollout_results)
        if hit_limit:
            n_hit_length_limit += 1

        ablation_results.append({
            "item_idx": task["item_idx"],
            "step_idx": task["step_idx"],
            "problem_id": task["problem_id"],
            "step_text_preview": task["step_text"][:300],
            "entropy_score": task["entropy_score"],
            "is_high_entropy": task["is_high_entropy"],
            "position": task["position"],
            "n_tokens": task["n_tokens"],
            "prefix_token_len": prefix_tl,
            "avg_gen_len": avg_gen_len,
            "max_gen_len": max_gen_len,
            "hit_length_limit": hit_limit,
            "n_correct": n_correct,
            "n_total": args.num_rollouts,
            "pns": pns,
            "rollout_details": rollout_results,
        })

    # Print length summary
    all_prefix_lens = [r["prefix_token_len"] for r in ablation_results]
    all_avg_gen_lens = [r["avg_gen_len"] for r in ablation_results]
    print(f"\n  Length summary:")
    print(f"    Prefix tokens: mean={np.mean(all_prefix_lens):.0f}, "
          f"max={max(all_prefix_lens)}")
    print(f"    Gen tokens (avg per task): mean={np.mean(all_avg_gen_lens):.0f}, "
          f"max={max(all_avg_gen_lens):.0f}")
    print(f"    Tasks hitting max_tokens limit: {n_hit_length_limit}/{len(ablation_results)} "
          f"({n_hit_length_limit/len(ablation_results)*100:.1f}%)")

    with open(phase3_path, "w") as f:
        json.dump(ablation_results, f, ensure_ascii=False)
    print(f"  Saved {len(ablation_results)} ablation results to {phase3_path}")

    del llm, all_outputs
    gc.collect()
    torch.cuda.empty_cache()

    return ablation_results


# ──────────────────────────── Analysis ────────────────────────────

def phase4_analysis(args, step_data, ablation_results, analysis_path):
    """Phase 4: Compute and display PNS analysis."""

    print(f"\n{'='*60}")
    print(f"Phase 4: PNS Analysis")
    print(f"{'='*60}")

    if len(ablation_results) == 0:
        print("  No results to analyze!")
        return

    pns_arr = np.array([d["pns"] for d in ablation_results])
    ent_arr = np.array([d["entropy_score"] for d in ablation_results])
    he_mask = np.array([d["is_high_entropy"] for d in ablation_results])

    # ── Basic PNS statistics ──
    analysis = {
        "config": {
            "model": str(args.model),
            "benchmark": args.benchmark,
            "ablation_mode": args.ablation_mode,
            "num_rollouts": args.num_rollouts,
            "entropy_top_ratio": args.entropy_top_ratio,
            "step_entropy_top_ratio": args.step_entropy_top_ratio,
            "test_all_steps": args.test_all_steps,
        },
        "summary": {
            "num_problems_with_correct": len(step_data),
            "num_steps_tested": len(ablation_results),
            "num_high_entropy_tested": int(he_mask.sum()),
            "num_low_entropy_tested": int((~he_mask).sum()),
        },
        "pns_stats": {
            "mean": float(pns_arr.mean()),
            "std": float(pns_arr.std()),
            "median": float(np.median(pns_arr)),
            "min": float(pns_arr.min()),
            "max": float(pns_arr.max()),
            "q25": float(np.percentile(pns_arr, 25)),
            "q75": float(np.percentile(pns_arr, 75)),
        },
        "pns_distribution": {
            "pns=0.0 (not necessary)": int((pns_arr == 0.0).sum()),
            "0.0<pns≤0.2": int(((pns_arr > 0.0) & (pns_arr <= 0.2)).sum()),
            "0.2<pns≤0.4": int(((pns_arr > 0.2) & (pns_arr <= 0.4)).sum()),
            "0.4<pns≤0.6": int(((pns_arr > 0.4) & (pns_arr <= 0.6)).sum()),
            "0.6<pns≤0.8": int(((pns_arr > 0.6) & (pns_arr <= 0.8)).sum()),
            "0.8<pns<1.0": int(((pns_arr > 0.8) & (pns_arr < 1.0)).sum()),
            "pns=1.0 (always necessary)": int((pns_arr == 1.0).sum()),
        },
    }

    # ── High-entropy vs Low-entropy PNS comparison ──
    if args.test_all_steps and he_mask.any() and (~he_mask).any():
        he_pns = pns_arr[he_mask]
        le_pns = pns_arr[~he_mask]
        analysis["high_vs_low_entropy"] = {
            "high_entropy_mean_pns": float(he_pns.mean()),
            "high_entropy_median_pns": float(np.median(he_pns)),
            "low_entropy_mean_pns": float(le_pns.mean()),
            "low_entropy_median_pns": float(np.median(le_pns)),
            "delta_mean": float(he_pns.mean() - le_pns.mean()),
            "high_entropy_pns_ge_0.6_ratio": float((he_pns >= 0.6).mean()),
            "low_entropy_pns_ge_0.6_ratio": float((le_pns >= 0.6).mean()),
        }

    # ── Entropy-PNS correlation ──
    if len(pns_arr) > 2:
        corr = float(np.corrcoef(ent_arr, pns_arr)[0, 1])
        analysis["entropy_pns_correlation"] = corr if not np.isnan(corr) else 0.0

    # ── Position analysis ──
    position_stats = {}
    for pos in ["early", "middle", "late"]:
        mask = np.array([d["position"] == pos for d in ablation_results])
        if mask.any():
            pos_pns = pns_arr[mask]
            position_stats[pos] = {
                "count": int(mask.sum()),
                "mean_pns": float(pos_pns.mean()),
                "median_pns": float(np.median(pos_pns)),
                "pns_ge_0.6_ratio": float((pos_pns >= 0.6).mean()),
            }
    analysis["position_stats"] = position_stats

    # ── Per-problem stats ──
    problem_pns = {}
    for d in ablation_results:
        pid = d["problem_id"]
        if pid not in problem_pns:
            problem_pns[pid] = []
        problem_pns[pid].append(d["pns"])
    analysis["per_problem_avg_pns"] = {
        pid: float(np.mean(vals)) for pid, vals in problem_pns.items()
    }

    # ── Top necessary / unnecessary steps ──
    sorted_by_pns = sorted(ablation_results, key=lambda x: x["pns"], reverse=True)
    analysis["top_20_most_necessary"] = [
        {
            "problem_id": d["problem_id"],
            "step_idx": d["step_idx"],
            "pns": d["pns"],
            "entropy_score": d["entropy_score"],
            "is_high_entropy": d["is_high_entropy"],
            "position": d["position"],
            "step_text": d["step_text_preview"][:200],
        }
        for d in sorted_by_pns[:20]
    ]
    analysis["top_20_least_necessary"] = [
        {
            "problem_id": d["problem_id"],
            "step_idx": d["step_idx"],
            "pns": d["pns"],
            "entropy_score": d["entropy_score"],
            "is_high_entropy": d["is_high_entropy"],
            "position": d["position"],
            "step_text": d["step_text_preview"][:200],
        }
        for d in sorted_by_pns[-20:]
    ]

    # Save
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    model_name = Path(args.model).name
    print(f"\n{'─'*60}")
    print(f"  Model:              {model_name}")
    print(f"  Benchmark:          {args.benchmark}")
    print(f"  Ablation mode:      {args.ablation_mode}")
    print(f"  Problems solved:    {len(step_data)}")
    print(f"  Steps tested:       {len(ablation_results)}")
    print(f"  Rollouts per step:  {args.num_rollouts}")
    print(f"{'─'*60}")
    print(f"  PNS Statistics:")
    for k, v in analysis["pns_stats"].items():
        print(f"    {k:>10s}: {v:.4f}")
    print(f"{'─'*60}")
    print(f"  PNS Distribution:")
    for label, count in analysis["pns_distribution"].items():
        pct = count / len(ablation_results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:35s}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"{'─'*60}")

    if "entropy_pns_correlation" in analysis:
        print(f"  Entropy↔PNS correlation: {analysis['entropy_pns_correlation']:.4f}")

    if "high_vs_low_entropy" in analysis:
        hv = analysis["high_vs_low_entropy"]
        print(f"{'─'*60}")
        print(f"  High-entropy steps mean PNS:  {hv['high_entropy_mean_pns']:.4f}")
        print(f"  Low-entropy steps mean PNS:   {hv['low_entropy_mean_pns']:.4f}")
        print(f"  Delta (HE - LE):              {hv['delta_mean']:+.4f}")
        print(f"  HE steps with PNS≥0.6:        {hv['high_entropy_pns_ge_0.6_ratio']:.1%}")
        print(f"  LE steps with PNS≥0.6:        {hv['low_entropy_pns_ge_0.6_ratio']:.1%}")

    if position_stats:
        print(f"{'─'*60}")
        print(f"  PNS by step position:")
        for pos, stats in position_stats.items():
            print(f"    {pos:8s}: mean={stats['mean_pns']:.3f}, "
                  f"median={stats['median_pns']:.3f}, "
                  f"PNS≥0.6: {stats['pns_ge_0.6_ratio']:.1%} "
                  f"(n={stats['count']})")

    print(f"{'─'*60}")
    print(f"  Results saved to: {args.output_dir}")

    # ── Key Insight ──
    print(f"\n{'='*60}")
    print("  Insights for PNS-DAPO Training:")
    print(f"{'='*60}")
    high_pns = (pns_arr >= 0.6).sum()
    low_pns = (pns_arr <= 0.2).sum()
    print(f"  • {high_pns}/{len(pns_arr)} ({high_pns/len(pns_arr)*100:.1f}%) steps "
          f"have PNS ≥ 0.6 (truly necessary)")
    print(f"  • {low_pns}/{len(pns_arr)} ({low_pns/len(pns_arr)*100:.1f}%) steps "
          f"have PNS ≤ 0.2 (removing them barely hurts)")
    if "entropy_pns_correlation" in analysis:
        corr = analysis["entropy_pns_correlation"]
        if corr > 0.3:
            print(f"  • Entropy-PNS correlation ({corr:.3f}) is moderate/strong →")
            print(f"    Entropy is a reasonable proxy for step importance.")
            print(f"    PNS can provide refined signal on top of entropy.")
        elif corr > 0.1:
            print(f"  • Entropy-PNS correlation ({corr:.3f}) is weak →")
            print(f"    PNS captures different information than entropy.")
            print(f"    PNS-based rewards could provide substantial new signal.")
        else:
            print(f"  • Entropy-PNS correlation ({corr:.3f}) is very weak →")
            print(f"    Entropy and PNS measure different things.")
            print(f"    PNS could significantly complement entropy-based methods.")
    print()

    return analysis


# ──────────────────────────── Main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PNS (Process Necessity Score) Estimation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (e.g. merged HF model)")
    parser.add_argument("--benchmark", default="math500",
                        choices=list(BENCHMARK_LOADERS.keys()),
                        help="Benchmark dataset to use")
    parser.add_argument("--num_problems", type=int, default=50,
                        help="Number of problems to use (0 = all)")
    parser.add_argument("--num_initial_samples", type=int, default=16,
                        help="Samples per problem to find correct responses")
    parser.add_argument("--num_rollouts", type=int, default=5,
                        help="Number of ablation rollouts per step")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max response tokens for generation")
    parser.add_argument("--entropy_top_ratio", type=float, default=0.2,
                        help="Token-level entropy top ratio for step scoring")
    parser.add_argument("--step_entropy_top_ratio", type=float, default=0.2,
                        help="Step-level top ratio for selecting high-entropy steps")
    parser.add_argument("--ablation_mode", choices=["counterfactual", "truncate", "skip"],
                        default="counterfactual",
                        help="Ablation strategy: counterfactual (original PNS paper - "
                             "instruct model not to use the target step), "
                             "truncate (cut before step), or "
                             "skip (remove step from middle)")
    parser.add_argument("--test_all_steps", action="store_true",
                        help="Test ALL steps (not just high-entropy) for "
                             "correlation analysis")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto from model name)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run (skip completed phases)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    model_name = Path(args.model).name
    if args.output_dir is None:
        args.output_dir = f"/ssdwork/fuzhizhang/pns_results/{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # File paths
    phase1_path = os.path.join(args.output_dir, "phase1_correct_responses.json")
    phase2_path = os.path.join(args.output_dir, "phase2_entropy_steps.json")
    phase3_path = os.path.join(args.output_dir, "phase3_ablation_rollouts.json")
    analysis_path = os.path.join(args.output_dir, "pns_analysis.json")

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # ── Phase 1 ──
    if args.resume and os.path.exists(phase1_path):
        print(f"\n[Phase 1] Loading cached: {phase1_path}")
        with open(phase1_path) as f:
            correct_data = json.load(f)
        print(f"  {len(correct_data)} correct responses loaded")
    else:
        correct_data = phase1_generate_correct(args, phase1_path)

    if len(correct_data) == 0:
        print("ERROR: No correct responses found! Check model or benchmark.")
        return

    # ── Phase 2 ──
    if args.resume and os.path.exists(phase2_path):
        print(f"\n[Phase 2] Loading cached: {phase2_path}")
        with open(phase2_path) as f:
            step_data = json.load(f)
        print(f"  {len(step_data)} items with step analysis loaded")
    else:
        step_data = phase2_entropy_steps(args, correct_data, phase2_path)

    # ── Phase 3 ──
    if args.resume and os.path.exists(phase3_path):
        print(f"\n[Phase 3] Loading cached: {phase3_path}")
        with open(phase3_path) as f:
            ablation_results = json.load(f)
        print(f"  {len(ablation_results)} ablation results loaded")
    else:
        ablation_results = phase3_ablation_rollouts(args, step_data, phase3_path)

    # ── Phase 4: Analysis ──
    phase4_analysis(args, step_data, ablation_results, analysis_path)


if __name__ == "__main__":
    main()
