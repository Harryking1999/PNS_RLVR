"""
PNS (Process Necessity Score) utilities for online PNS-based PRM training.

This module implements the real PNS computation pipeline that runs during
training, as an auxiliary reward integrated into the advantage function.

Pipeline (called between advantage computation and actor update):
  1. Select high-entropy steps from the current batch (reuse existing masks).
  2. Build counterfactual prompts for selected steps.
  3. Run ablation rollouts via the existing rollout engine.
  4. Score ablation rollouts to get true PNS values.
  5. Inject PNS bonus into the advantage tensor.

PNS definition:
  - Positive sample (acc=1): PNS = 1 - n_correct_ablation / N
    Bonus = lambda * (PNS - 0.5)   [shift so PNS<0.5 gets punished]
  - Negative sample (acc=0): PNS = -(n_correct_ablation / N)
    Bonus = lambda * PNS           [no shift; PNS in [-1, 0]]
"""

import logging
import multiprocessing
import os
import queue
import time
from typing import Optional

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__name__)

_SCORE_TIMEOUT_COUNT = 0


def _pns_score_worker(result_queue, compute_score_fn, data_source, solution_str, ground_truth):
    """Top-level worker function for process-based scoring timeout.

    Must be defined at module level so it is picklable for all multiprocessing
    start methods (fork/forkserver/spawn).
    """
    try:
        result = compute_score_fn(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info={},
        )
        result_queue.put((True, result))
    except Exception as e:
        result_queue.put((False, repr(e)))


def compute_score_with_timeout(compute_score_fn, data_source, solution_str, ground_truth, timeout=15):
    """Call compute_score_fn with a process-based timeout.

    Previous implementation used a daemon thread, which had two critical flaws:
      1. Python threads cannot be killed — a timed-out thread stays alive forever,
         leaking memory and potentially spawning child processes in the background.
      2. signal.SIGALRM (used by naive_dapo.are_equal_under_sympy's @timeout
         decorator) only works in the main thread. In a daemon thread it raises
         ValueError, silently breaking the sympy timeout and causing incorrect
         scores (not just timeouts).

    This version uses multiprocessing.Process:
      - The child process CAN be killed (SIGTERM/SIGKILL) on timeout.
      - signal.SIGALRM works in the child (it is the child's main thread).
      - All memory allocated by the child is freed by the OS on kill.
      - Detailed diagnostics are logged on timeout for debugging.
    """
    global _SCORE_TIMEOUT_COUNT

    q = multiprocessing.Queue(maxsize=1)
    proc = multiprocessing.Process(
        target=_pns_score_worker,
        args=(q, compute_score_fn, data_source, solution_str, ground_truth),
    )
    t0 = time.monotonic()
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # ── Timeout: collect diagnostics BEFORE killing ──
        _SCORE_TIMEOUT_COUNT += 1
        diag = ""
        try:
            import psutil

            parent_rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            child_rss = psutil.Process(proc.pid).memory_info().rss / (1024**3)
            sys_mem = psutil.virtual_memory()
            diag = (
                f"parent_rss={parent_rss:.1f}GB "
                f"child_rss={child_rss:.1f}GB "
                f"sys_used={sys_mem.used / (1024**3):.1f}GB/"
                f"{sys_mem.total / (1024**3):.1f}GB "
                f"sys_avail={sys_mem.available / (1024**3):.1f}GB"
            )
        except Exception:
            diag = "(diagnostics unavailable)"

        logger.warning(
            f"[PNS] compute_score timed out after {timeout}s "
            f"(total timeouts: {_SCORE_TIMEOUT_COUNT}), marking as incorrect. "
            f"data_source={data_source!r} "
            f"solution_len={len(solution_str)} "
            f"gt={ground_truth!r:.80s} "
            f"{diag}"
        )

        # Kill the child process
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
            if proc.is_alive():
                logger.error(f"[PNS] Scoring child process {proc.pid} did not die after SIGKILL!")
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
        return 0.0

    # ── Process finished within timeout ──
    elapsed = time.monotonic() - t0
    try:
        success, result = q.get(timeout=0.5)
    except queue.Empty:
        exitcode = proc.exitcode
        logger.warning(
            f"[PNS] compute_score child exited (code={exitcode}) but returned no result "
            f"after {elapsed:.1f}s, marking as incorrect. "
            f"data_source={data_source!r} solution_len={len(solution_str)}"
        )
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
        return 0.0

    try:
        q.close()
        q.join_thread()
    except Exception:
        pass

    if not success:
        # result contains the repr of the exception from the child
        return 0.0

    # Log slow (but successful) scoring calls for awareness
    if elapsed > 5.0:
        logger.info(
            f"[PNS] compute_score slow but succeeded in {elapsed:.1f}s "
            f"(data_source={data_source!r}, solution_len={len(solution_str)})"
        )

    return result

# ──────────────── Counterfactual prompt templates ────────────────

COUNTERFACTUAL_MID_TEMPLATE = (
    "You have been working on this problem and have made progress as shown above. "
    "However, for the next step, you must NOT use the following reasoning:\n"
    '"{FORBIDDEN}"\n\n'
    "Please find an alternative approach to continue solving the problem. "
    "Do NOT repeat or rephrase the forbidden step. "
    "Output the final answer within \\boxed{}."
)

COUNTERFACTUAL_FIRST_STEP_TEMPLATE = (
    "When solving this problem, do NOT begin with or use the following reasoning approach:\n"
    '"{FORBIDDEN}"\n\n'
    "Find a different way to start solving the problem. "
    "Do NOT repeat or rephrase the forbidden approach. "
    "Output the final answer within \\boxed{}."
)


# ──────────────── Step segmentation ────────────────

SENTENCE_ENDINGS = {'.', '?', '!', '。', '？', '！'}


def precompute_step_boundary_lookup(tokenizer) -> torch.Tensor:
    """Pre-compute boolean lookup: boundary_lookup[token_id] = True if token
    ends a sentence (newline, period, question mark, etc.)."""
    import time
    t0 = time.time()
    total_ids = len(tokenizer)
    boundary_lookup = torch.zeros(total_ids, dtype=torch.bool)

    for token_id in range(total_ids):
        try:
            text = tokenizer.decode([token_id])
            if not text:
                continue
            if '\n' in text:
                boundary_lookup[token_id] = True
                continue
            stripped = text.rstrip()
            if stripped and stripped[-1] in SENTENCE_ENDINGS:
                boundary_lookup[token_id] = True
        except Exception:
            pass

    elapsed = time.time() - t0
    n_boundary = boundary_lookup.sum().item()
    logger.info(
        f"[PNS] Pre-computed {n_boundary}/{total_ids} boundary token IDs in {elapsed:.1f}s"
    )
    return boundary_lookup


def segment_response_into_steps(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    boundary_lookup: torch.Tensor,
) -> list[tuple[int, int]]:
    """Segment a single response into steps based on sentence boundaries.

    Args:
        response_ids: [S] token ids of the response part.
        response_mask: [S] binary mask (1 = valid response token).
        boundary_lookup: [V] boolean lookup for boundary tokens.
            Must already be on the same device as response_ids.

    Returns:
        List of (start, end) index pairs for each step.
    """
    valid = response_mask.bool()
    if not valid.any():
        return []

    lookup_size = boundary_lookup.shape[0]
    ids_safe = response_ids.clamp(0, lookup_size - 1)
    is_boundary = boundary_lookup[ids_safe] & valid

    valid_positions = valid.nonzero(as_tuple=False).squeeze(1)
    if valid_positions.numel() == 0:
        return []
    resp_start = valid_positions[0].item()
    resp_end = valid_positions[-1].item() + 1

    boundary_positions = is_boundary.nonzero(as_tuple=False).squeeze(1)

    steps = []
    step_start = resp_start
    if boundary_positions.numel() > 0:
        for bp_idx in range(boundary_positions.numel()):
            bp = boundary_positions[bp_idx].item()
            if bp >= step_start:
                steps.append((step_start, bp + 1))
                step_start = bp + 1
    if step_start < resp_end:
        steps.append((step_start, resp_end))

    return steps


# ──────────────── Select high-entropy steps for PNS ────────────────

def select_pns_steps(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
    boundary_lookup: torch.Tensor,
    token_top_ratio: float,
    step_top_ratio: float,
    token_top_scope: str,
    acc: np.ndarray,
    max_steps_per_batch: int,
) -> list[dict]:
    """Select high-entropy steps for PNS ablation testing.

    This reuses the same step scoring logic as Step Forking DAPO but selects
    steps from the entire batch and limits total count for efficiency.

    Args:
        entropy: [B, S] per-token entropy from forward pass.
        response_mask: [B, S] response mask.
        response_ids: [B, S] response token ids.
        boundary_lookup: [V] boundary lookup table.
        token_top_ratio: fraction of tokens to select as high-entropy.
        step_top_ratio: fraction of steps to select per rollout.
        token_top_scope: "rollout" or "batch".
        acc: [B] accuracy array (0 or 1).
        max_steps_per_batch: max number of steps to test in the entire batch.

    Returns:
        List of dicts, each with keys:
            - rollout_idx: index in the batch
            - step_start, step_end: token indices of the step
            - step_text_tokens: token ids of the step
            - prefix_token_ids: token ids of the prefix (before step)
            - entropy_score: sum of high-entropy token entropies in this step
            - acc: accuracy of this rollout (0 or 1)
    """
    from verl.trainer.ppo.core_algos import (
        get_global_entropy_top_mask,
        get_rollout_entropy_top_mask,
    )

    B, S = entropy.shape
    device = entropy.device

    # Step 1: get token-level entropy mask
    if token_top_scope == "batch":
        token_top_mask = get_global_entropy_top_mask(
            entropy=entropy, response_mask=response_mask, top_ratio=token_top_ratio
        )
    elif token_top_scope == "rollout":
        token_top_mask = get_rollout_entropy_top_mask(
            entropy=entropy, response_mask=response_mask, top_ratio=token_top_ratio
        )
    else:
        raise ValueError(f"Unsupported token_top_scope={token_top_scope}")

    boundary_lookup = boundary_lookup.to(device)

    # Step 2: for each rollout, segment into steps and score them
    all_candidates = []
    for b in range(B):
        steps = segment_response_into_steps(
            response_ids=response_ids[b],
            response_mask=response_mask[b],
            boundary_lookup=boundary_lookup,
        )
        if len(steps) == 0:
            continue

        # Score each step
        step_scores = []
        for si, (s_start, s_end) in enumerate(steps):
            step_valid = response_mask[b, s_start:s_end].bool()
            if not step_valid.any():
                step_scores.append(0.0)
                continue
            selected_in_step = token_top_mask[b, s_start:s_end].bool() & step_valid
            if selected_in_step.any():
                score = entropy[b, s_start:s_end][selected_in_step].sum().item()
            else:
                score = 0.0
            step_scores.append(score)

        # Select top steps within this rollout
        num_selected = max(1, int(len(steps) * step_top_ratio + 0.9999))
        num_selected = min(num_selected, len(steps))
        topk_indices = sorted(
            range(len(steps)),
            key=lambda i: step_scores[i],
            reverse=True,
        )[:num_selected]

        for idx in topk_indices:
            s_start, s_end = steps[idx]
            # Skip tiny steps (< 3 tokens)
            n_tokens = s_end - s_start
            if n_tokens < 3:
                continue
            all_candidates.append({
                "rollout_idx": b,
                "step_idx_in_rollout": idx,
                "step_start": s_start,
                "step_end": s_end,
                "n_tokens": n_tokens,
                "entropy_score": step_scores[idx],
                "acc": float(acc[b]) if b < len(acc) else 0.0,
            })

    # Step 3: globally select top candidates by entropy_score
    if len(all_candidates) > max_steps_per_batch:
        all_candidates.sort(key=lambda c: c["entropy_score"], reverse=True)
        all_candidates = all_candidates[:max_steps_per_batch]

    return all_candidates


# ──────────────── Build counterfactual prompts ────────────────

def build_counterfactual_prompts(
    candidates: list[dict],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer,
    max_prompt_len: int,
) -> list[dict]:
    """Build counterfactual ablation prompts for PNS testing.

    For each candidate step, constructs a multi-turn prompt that tells the
    model to avoid using that step and find an alternative approach.

    Args:
        candidates: list from select_pns_steps.
        input_ids: [B, total_len] full input_ids (prompt + response).
        attention_mask: [B, total_len] attention mask.
        response_ids: [B, resp_len] response token ids.
        tokenizer: tokenizer for decoding.
        max_prompt_len: maximum allowed prompt length in tokens (skip if longer).

    Returns:
        Filtered list of candidates with added 'ablation_prompt_ids' key.
    """
    results = []
    n_skipped_long = 0

    for cand in candidates:
        b = cand["rollout_idx"]

        # Get prompt part: everything before response
        attn = attention_mask[b]
        prompt_len = attn.shape[0] - response_ids.shape[1]
        # The actual prompt (non-padding)
        valid_prompt_start = (attn[:prompt_len] == 1).nonzero(as_tuple=False)
        if valid_prompt_start.numel() == 0:
            continue
        valid_prompt_ids = input_ids[b, valid_prompt_start[0].item():prompt_len]

        # Decode the original problem (user prompt)
        problem_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

        # Decode the step text (forbidden text)
        step_token_ids = response_ids[b, cand["step_start"]:cand["step_end"]]
        forbidden_text = tokenizer.decode(step_token_ids, skip_special_tokens=True).strip()

        if not forbidden_text:
            continue

        # Decode the prefix (response before this step)
        prefix_token_ids = response_ids[b, :cand["step_start"]]
        prefix_text = tokenizer.decode(prefix_token_ids, skip_special_tokens=True)

        # Build counterfactual prompt
        if cand["step_start"] == 0:
            # First step: single user turn
            counterfactual_msg = COUNTERFACTUAL_FIRST_STEP_TEMPLATE.replace(
                "{FORBIDDEN}", forbidden_text
            )
            messages = [
                {"role": "user", "content": problem_text + "\n\n" + counterfactual_msg},
            ]
        else:
            # Mid-solution: multi-turn
            counterfactual_msg = COUNTERFACTUAL_MID_TEMPLATE.replace(
                "{FORBIDDEN}", forbidden_text
            )
            messages = [
                {"role": "user", "content": problem_text},
                {"role": "assistant", "content": prefix_text},
                {"role": "user", "content": counterfactual_msg},
            ]

        ablation_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ablation_prompt_ids = tokenizer.encode(ablation_prompt, add_special_tokens=False)

        # Check length
        if len(ablation_prompt_ids) > max_prompt_len:
            n_skipped_long += 1
            continue

        cand["ablation_prompt_ids"] = ablation_prompt_ids
        cand["ablation_prompt_len"] = len(ablation_prompt_ids)
        results.append(cand)

    if n_skipped_long > 0:
        logger.info(f"[PNS] Skipped {n_skipped_long} steps with prompt > {max_prompt_len} tokens")

    return results


# ──────────────── Prepare DataProto for ablation rollouts ────────────────

def prepare_ablation_batch(
    candidates: list[dict],
    tokenizer,
    max_prompt_len: int,
    device: str = "cpu",
) -> Optional[DataProto]:
    """Pack ablation prompts into a DataProto for generate_sequences.

    The DataProto format matches what vLLM rollout expects:
      - input_ids: [N, max_prompt_len] left-padded
      - attention_mask: [N, max_prompt_len]
      - position_ids: [N, max_prompt_len]

    Args:
        candidates: list with 'ablation_prompt_ids' key.
        tokenizer: tokenizer.
        max_prompt_len: max prompt length for padding.
        device: device for tensors.

    Returns:
        DataProto ready for generate_sequences, or None if no candidates.
    """
    if not candidates:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Find max length in this batch of prompts
    actual_max_len = max(c["ablation_prompt_len"] for c in candidates)
    # Pad to this length (left-pad for decoder-only models)
    padded_len = actual_max_len

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []

    for cand in candidates:
        prompt_ids = cand["ablation_prompt_ids"]
        n_pad = padded_len - len(prompt_ids)

        input_ids = [pad_token_id] * n_pad + prompt_ids
        attn_mask = [0] * n_pad + [1] * len(prompt_ids)
        pos_ids = [0] * n_pad + list(range(len(prompt_ids)))

        all_input_ids.append(input_ids)
        all_attention_mask.append(attn_mask)
        all_position_ids.append(pos_ids)

    batch_dict = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long, device=device),
        "position_ids": torch.tensor(all_position_ids, dtype=torch.long, device=device),
    }

    data = DataProto.from_single_dict(batch_dict)
    return data


# ──────────────── Score ablation rollouts ────────────────

def score_ablation_rollouts(
    ablation_output: DataProto,
    candidates: list[dict],
    tokenizer,
    compute_score_fn,
    num_rollouts: int,
) -> list[dict]:
    """Score ablation rollout outputs and compute PNS for each candidate step.

    Args:
        ablation_output: DataProto from generate_sequences with ablation rollouts.
            Contains responses for all candidates × num_rollouts.
        candidates: original candidate list with metadata.
        tokenizer: tokenizer for decoding.
        compute_score_fn: low-level scoring function with signature
            (data_source, solution_str, ground_truth, extra_info) -> float|dict.
            This is RewardManager.compute_score, NOT the RewardManager itself.
        num_rollouts: N rollouts per ablation task.

    Returns:
        candidates list updated with 'pns_value' and 'pns_bonus' keys.
    """
    responses = ablation_output.batch["responses"]
    n_total = responses.shape[0]
    n_candidates = len(candidates)

    # ablation_output has n_candidates * num_rollouts rows
    assert n_total == n_candidates * num_rollouts, (
        f"Expected {n_candidates * num_rollouts} rows, got {n_total}"
    )

    # generate_sequences does NOT return response_mask directly;
    # compute it from attention_mask (prompt_mask ++ response_mask).
    response_length = responses.size(1)
    attention_mask = ablation_output.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # Batch-decode all responses at once for efficiency
    valid_id_lists = []
    for ri in range(n_total):
        valid_len = int(response_mask[ri].sum().item())
        valid_id_lists.append(responses[ri][:valid_len])
    all_response_texts = tokenizer.batch_decode(valid_id_lists, skip_special_tokens=True)
    eos_token = tokenizer.eos_token
    if eos_token:
        all_response_texts = [
            t[: -len(eos_token)] if t.endswith(eos_token) else t
            for t in all_response_texts
        ]

    for ci, cand in enumerate(candidates):
        start_row = ci * num_rollouts
        end_row = start_row + num_rollouts

        ground_truth = cand.get("ground_truth", "")
        data_source = cand.get("data_source", "")

        n_correct = 0
        for ri in range(start_row, end_row):
            response_text = all_response_texts[ri]
            try:
                result = compute_score_with_timeout(
                    compute_score_fn, data_source, response_text, ground_truth, timeout=15
                )
                if isinstance(result, dict):
                    score = result.get("score", result.get("acc", 0.0))
                else:
                    score = float(result)
                if score > 0:
                    n_correct += 1
            except Exception:
                pass

        acc = cand["acc"]
        if acc > 0:
            # Positive sample: PNS = 1 - n_correct/N
            pns = 1.0 - n_correct / num_rollouts
            # Shift so PNS < 0.5 is punished, PNS > 0.5 is rewarded
            bonus = pns - 0.5
        else:
            # Negative sample: PNS = -(n_correct/N)
            pns = -(n_correct / num_rollouts)
            bonus = pns

        cand["pns_value"] = pns
        cand["pns_bonus"] = bonus
        cand["n_correct_ablation"] = n_correct

    return candidates


# ──────────────── Inject PNS into advantage ────────────────

def inject_pns_into_advantage(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    candidates: list[dict],
    pns_lambda: float,
    dry_run: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Add PNS bonus to the advantage tensor for selected steps.

    For each candidate step, adds pns_lambda * pns_bonus to all tokens
    in that step's range. In dry_run mode, metrics are computed but
    advantages are NOT modified.

    Args:
        advantages: [B, S] advantage tensor (will be modified in-place).
        response_mask: [B, S] response mask.
        candidates: list with 'pns_value', 'pns_bonus' keys.
        pns_lambda: scaling factor for PNS bonus.
        dry_run: if True, compute metrics but do not modify advantages.

    Returns:
        Tuple of (modified advantages, metrics dict).
    """
    metrics = {}
    if not candidates:
        return advantages, metrics

    pns_values = []
    pns_bonuses = []
    pos_pns = []
    pos_bonuses = []
    neg_pns = []
    neg_bonuses = []

    for cand in candidates:
        b = cand["rollout_idx"]
        s_start = cand["step_start"]
        s_end = cand["step_end"]
        bonus = cand["pns_bonus"]
        pns = cand["pns_value"]

        # Add bonus to all tokens in this step (skip in dry_run)
        if not dry_run and pns_lambda > 0:
            step_mask = response_mask[b, s_start:s_end].float()
            advantages[b, s_start:s_end] += pns_lambda * bonus * step_mask

        pns_values.append(pns)
        pns_bonuses.append(bonus)
        if cand["acc"] > 0:
            pos_pns.append(pns)
            pos_bonuses.append(bonus)
        else:
            neg_pns.append(pns)
            neg_bonuses.append(bonus)

    # Compute metrics
    metrics["pns/n_tested_steps"] = len(candidates)
    metrics["pns/mean_pns"] = float(np.mean(pns_values))
    metrics["pns/mean_bonus"] = float(np.mean(pns_bonuses))
    metrics["pns/dry_run"] = 1.0 if dry_run else 0.0
    if pos_pns:
        metrics["pns/mean_pos_pns"] = float(np.mean(pos_pns))
        metrics["pns/mean_pos_bonus"] = float(np.mean(pos_bonuses))
        metrics["pns/n_pos_steps"] = len(pos_pns)
    if neg_pns:
        metrics["pns/mean_neg_pns"] = float(np.mean(neg_pns))
        metrics["pns/mean_neg_bonus"] = float(np.mean(neg_bonuses))
        metrics["pns/n_neg_steps"] = len(neg_pns)

    return advantages, metrics


# ──────────────── Main PNS computation orchestrator ────────────────

def compute_pns_for_batch(
    batch: DataProto,
    entropy: torch.Tensor,
    boundary_lookup: torch.Tensor,
    tokenizer,
    compute_score_fn,
    generate_fn,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    """Main PNS computation function called from ray_trainer.

    Orchestrates the full PNS pipeline:
      1. Select high-entropy steps
      2. Build counterfactual prompts
      3. Run ablation rollouts
      4. Score and compute PNS
      5. Inject into advantage

    Args:
        batch: full training batch DataProto with advantages already computed.
        entropy: [B, S] entropy tensor from old_log_prob computation.
        boundary_lookup: [V] pre-computed boundary lookup.
        tokenizer: tokenizer.
        compute_score_fn: low-level scoring function with signature
            (data_source, solution_str, ground_truth, extra_info) -> float|dict.
            This is RewardManager.compute_score, NOT the RewardManager itself.
        generate_fn: callable that takes DataProto and returns DataProto
                     (wraps self.actor_rollout_wg.generate_sequences).
        config: dict with PNS configuration:
            - pns_lambda: float, scaling factor
            - pns_num_rollouts: int, number of ablation rollouts per step
            - pns_step_ratio: float, ratio of batch for PNS testing
            - token_top_ratio: float, for entropy token selection
            - step_top_ratio: float, for step selection per rollout
            - token_top_scope: str, "rollout" or "batch"
            - max_prompt_length: int, max prompt length from data config
            - max_response_length: int, for generation length limit
            - pns_ablation_batch_size: int, max prompts per ablation generate call

    Returns:
        Tuple of (modified advantages, pns_metrics dict).
    """
    pns_lambda = config.get("pns_lambda", 0.5)
    pns_num_rollouts = config.get("pns_num_rollouts", 5)
    pns_step_ratio = config.get("pns_step_ratio", 0.5)
    token_top_ratio = config.get("token_top_ratio", 0.2)
    step_top_ratio = config.get("step_top_ratio", 0.1)
    token_top_scope = config.get("token_top_scope", "rollout")
    max_prompt_length = config.get("max_prompt_length", 2048)
    max_response_length = config.get("max_response_length", 4096)
    ablation_batch_size = config.get("pns_ablation_batch_size", 64)
    dry_run = config.get("pns_dry_run", False)

    advantages = batch.batch["advantages"]
    response_mask = batch.batch["response_mask"]
    response_ids = batch.batch["responses"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]

    B, S = response_mask.shape

    # Get acc from non_tensor_batch
    if "acc" in batch.non_tensor_batch:
        acc = np.array(batch.non_tensor_batch["acc"], dtype=np.float32)
    else:
        logger.warning("[PNS] 'acc' not found in batch.non_tensor_batch, skipping PNS")
        return advantages, {}

    # Get ground_truth and data_source for scoring ablation rollouts
    ground_truths = batch.non_tensor_batch.get("reward_model", None)
    data_sources = batch.non_tensor_batch.get("data_source", None)

    # Calculate max steps to test
    max_steps_per_batch = int(B * pns_step_ratio)
    max_steps_per_batch = max(1, max_steps_per_batch)

    logger.info(f"[PNS] Selecting up to {max_steps_per_batch} steps from batch of {B} rollouts")

    # Step 1: Select high-entropy steps
    candidates = select_pns_steps(
        entropy=entropy,
        response_mask=response_mask,
        response_ids=response_ids,
        boundary_lookup=boundary_lookup,
        token_top_ratio=token_top_ratio,
        step_top_ratio=step_top_ratio,
        token_top_scope=token_top_scope,
        acc=acc,
        max_steps_per_batch=max_steps_per_batch,
    )

    if not candidates:
        logger.info("[PNS] No candidate steps selected, skipping PNS phase")
        return advantages, {"pns/n_tested_steps": 0}

    # Attach ground_truth and data_source to each candidate
    for cand in candidates:
        b = cand["rollout_idx"]
        if ground_truths is not None:
            gt = ground_truths[b]
            if isinstance(gt, dict):
                cand["ground_truth"] = gt.get("ground_truth", "")
            else:
                cand["ground_truth"] = str(gt)
        else:
            cand["ground_truth"] = ""
        if data_sources is not None:
            cand["data_source"] = str(data_sources[b])
        else:
            cand["data_source"] = ""

    logger.info(f"[PNS] Selected {len(candidates)} candidate steps for PNS testing")

    # Step 2: Build counterfactual prompts
    # Counterfactual prompt = original_prompt + prefix (partial response) + template.
    # Worst case ≈ max_prompt_length + max_response_length.
    # Add 2000 token buffer for chat template overhead and the intervention instruction.
    max_prompt_len = max_prompt_length + max_response_length + 2000
    candidates = build_counterfactual_prompts(
        candidates=candidates,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_ids=response_ids,
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
    )

    if not candidates:
        logger.info("[PNS] No candidates left after prompt construction, skipping")
        return advantages, {"pns/n_tested_steps": 0}

    logger.info(
        f"[PNS] {len(candidates)} candidates × {pns_num_rollouts} rollouts = "
        f"{len(candidates) * pns_num_rollouts} ablation generations"
    )

    # Step 3 & 4: Prepare DataProto and run ablation rollouts (batched for efficiency)
    all_ablation_outputs = []
    n_gen_failed = 0
    total_prompts = len(candidates)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    for batch_start in range(0, total_prompts, ablation_batch_size):
        batch_end = min(batch_start + ablation_batch_size, total_prompts)
        sub_candidates = candidates[batch_start:batch_end]
        sub_batch = prepare_ablation_batch(
            candidates=sub_candidates,
            tokenizer=tokenizer,
            max_prompt_len=max_prompt_len,
        )
        if sub_batch is None:
            continue

        sub_batch.meta_info["eos_token_id"] = eos_token_id
        sub_batch.meta_info["pad_token_id"] = pad_token_id
        sub_batch.meta_info["do_sample"] = True
        sub_batch.meta_info["pns_num_rollouts"] = pns_num_rollouts

        # Generate with n rollouts per prompt
        try:
            sub_output = generate_fn(sub_batch, pns_num_rollouts=pns_num_rollouts)
            all_ablation_outputs.append((sub_output, sub_candidates))
        except Exception as e:
            logger.warning(f"[PNS] Ablation generation failed: {e}")
            # Mark these candidates as untested (bonus=0, no effect on advantage)
            for cand in sub_candidates:
                cand["pns_value"] = 0.0
                cand["pns_bonus"] = 0.0
                cand["n_correct_ablation"] = 0
            n_gen_failed += len(sub_candidates)

    # Step 5: Score ablation rollouts and compute PNS
    scored_candidates = []
    for sub_output, sub_candidates in all_ablation_outputs:
        scored = score_ablation_rollouts(
            ablation_output=sub_output,
            candidates=sub_candidates,
            tokenizer=tokenizer,
            compute_score_fn=compute_score_fn,
            num_rollouts=pns_num_rollouts,
        )
        scored_candidates.extend(scored)

    # Include all candidates (including those that failed generation with pns=0)
    for cand in candidates:
        if "pns_value" in cand and cand not in scored_candidates:
            scored_candidates.append(cand)

    # Log per-candidate PNS details
    mode_str = "DRY-RUN" if dry_run else "ACTIVE"
    logger.info(f"[PNS {mode_str}] Per-candidate PNS details ({len(scored_candidates)} candidates):")
    for ci, cand in enumerate(scored_candidates):
        logger.info(
            f"  [{ci}] rollout={cand['rollout_idx']} "
            f"step=[{cand['step_start']}:{cand['step_end']}] "
            f"acc={cand['acc']:.0f} "
            f"n_correct={cand.get('n_correct_ablation', '?')}/{pns_num_rollouts} "
            f"PNS={cand.get('pns_value', 0):.3f} "
            f"bonus={cand.get('pns_bonus', 0):.3f}"
        )

    # Step 6: Inject PNS into advantage (skipped in dry_run mode)
    advantages, pns_metrics = inject_pns_into_advantage(
        advantages=advantages,
        response_mask=response_mask,
        candidates=scored_candidates,
        pns_lambda=pns_lambda,
        dry_run=dry_run,
    )

    if n_gen_failed > 0:
        pns_metrics["pns/n_gen_failed"] = n_gen_failed

    return advantages, pns_metrics
