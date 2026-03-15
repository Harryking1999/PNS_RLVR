#!/usr/bin/env python3
"""
Step-Level Entropy Visualization Tool

目标：验证「高熵 token」和「高熵步骤」之间是否有人类可理解的一致关系。

流程：
1. 加载 base 模型，对若干数学题生成 response
2. 计算每个 token 的熵
3. 按指定方法切分为步骤 (punctuation 或 double_newline)
4. 用不同超参 (10%, 20%, 30%) 展示被选中的步骤

Usage:
    # GPU 模式：生成 + 可视化 + 保存缓存
    CUDA_VISIBLE_DEVICES=3 python tools/visualize_step_entropy.py --save_cache tools/cache_1.7B.pkl
    CUDA_VISIBLE_DEVICES=3 python tools/visualize_step_entropy.py --model /path/to/Qwen3-4B-Base

    # CPU 模式：从缓存加载，换用 \\n\\n 切分（无需 GPU）
    python tools/visualize_step_entropy.py --load_cache tools/cache_1.7B.pkl --segment_method double_newline
"""

import os
import argparse
import json
import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ──────────────────────────── ANSI colors ────────────────────────────

class C:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def entropy_color(ent, max_ent):
    """Map entropy to a 256-color ANSI background: blue(low) -> yellow(mid) -> red(high)."""
    ratio = min(ent / (max_ent + 1e-6), 1.0)
    # 232-255 are grayscale in 256-color; use 16-231 color cube
    # Simple: ratio -> color index in the red spectrum
    if ratio < 0.33:
        return "\033[48;5;17m"   # dark blue bg
    elif ratio < 0.66:
        return "\033[48;5;136m"  # yellow/brown bg
    else:
        return "\033[48;5;124m"  # dark red bg


# ──────────────────────────── Core logic ─────────────────────────────

def generate_and_compute_entropy(model, tokenizer, prompt_text,
                                  max_new_tokens=2048, temperature=1.0,
                                  top_p=1.0, device="cuda:0"):
    """
    生成 response，并计算每个 response token 的熵。

    Returns:
        response_text: str
        response_tokens: list[str]  — 每个 token decode 后的文本
        response_token_ids: list[int]
        token_entropies: np.ndarray of shape (response_len,)
    """
    input_ids = tokenizer(str(prompt_text), return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0]          # (total_len,)
    response_ids = generated_ids[prompt_len:]     # response only

    if len(response_ids) == 0:
        return "", [], [], np.array([])

    # Forward pass to get logits
    with torch.no_grad():
        full_output = model(generated_ids.unsqueeze(0))
        logits = full_output.logits[0]            # (total_len, vocab_size)

    # logits[t] predicts token at position t+1
    response_logits = logits[prompt_len - 1 : prompt_len - 1 + len(response_ids)]  # (resp_len, V)

    # Entropy = -sum(p * log p)
    probs = torch.softmax(response_logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    token_entropies = -(probs * log_probs).sum(dim=-1).cpu().numpy()

    response_token_ids = response_ids.cpu().tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response_text, response_tokens, response_token_ids, token_entropies


def segment_into_steps_punctuation(response_tokens, response_token_ids):
    """
    将 response 按换行符 + 句末标点切分为步骤。

    切分规则：遇到分隔符 token 时，该 token 作为当前 step 的最后一个 token，
    下一个 token 开启新 step。

    分隔符：换行符 \\n,  句末标点 . ? ! 。 ？ ！
    """
    SENTENCE_ENDINGS = {'.', '?', '!', '。', '？', '！'}
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

    # last segment
    if current_start < len(response_tokens):
        steps.append({
            'start': current_start,
            'end': len(response_tokens),
            'token_indices': list(range(current_start, len(response_tokens))),
            'text': ''.join(response_tokens[current_start:]),
        })

    return steps


def segment_into_steps_double_newline(response_tokens, response_token_ids):
    """
    将 response 按 \\n\\n (连续两个换行) 切分为步骤。

    切分规则：在 token 序列中找到 \\n\\n 出现的位置，以此为分隔符切分。
    包含 \\n\\n 的 token 作为当前 step 的最后一个 token。
    """
    # 先拼出完整文本，记录每个 token 对应的字符区间
    char_offsets = []  # (char_start, char_end) for each token
    pos = 0
    full_text = ''.join(response_tokens)
    for i, tok in enumerate(response_tokens):
        char_offsets.append((pos, pos + len(tok)))
        pos += len(tok)

    # 找出所有 \n\n 在 full_text 中的位置
    split_positions = []
    search_start = 0
    while True:
        idx = full_text.find('\n\n', search_start)
        if idx == -1:
            break
        split_positions.append(idx)
        # 跳过连续的换行（如 \n\n\n 只算一次）
        search_start = idx + 2
        while search_start < len(full_text) and full_text[search_start] == '\n':
            search_start += 1

    if not split_positions:
        # 没有 \n\n，整个 response 为一个 step
        return [{
            'start': 0,
            'end': len(response_tokens),
            'token_indices': list(range(len(response_tokens))),
            'text': full_text,
        }]

    # 对每个 \n\n 位置，找到它落在哪个 token 里（作为该 step 的末尾 token）
    boundary_token_indices = []
    for sp in split_positions:
        # \n\n 的最后一个字符在 sp+1 处
        nn_end = sp + 1
        for i, (cs, ce) in enumerate(char_offsets):
            if cs <= nn_end < ce:
                boundary_token_indices.append(i)
                break
        else:
            # nn_end 正好在某个 token 末尾
            for i, (cs, ce) in enumerate(char_offsets):
                if ce == nn_end + 1:
                    boundary_token_indices.append(i)
                    break

    # 去重并排序
    boundary_token_indices = sorted(set(boundary_token_indices))

    # 构建 steps
    steps = []
    current_start = 0
    for boundary_idx in boundary_token_indices:
        end = boundary_idx + 1
        if end <= current_start:
            continue
        if end >= len(response_tokens):
            break
        steps.append({
            'start': current_start,
            'end': end,
            'token_indices': list(range(current_start, end)),
            'text': ''.join(response_tokens[current_start:end]),
        })
        current_start = end

    # last segment
    if current_start < len(response_tokens):
        steps.append({
            'start': current_start,
            'end': len(response_tokens),
            'token_indices': list(range(current_start, len(response_tokens))),
            'text': ''.join(response_tokens[current_start:]),
        })

    return steps


def segment_into_steps(response_tokens, response_token_ids, method="punctuation"):
    """根据 method 选择切分方法。"""
    if method == "double_newline":
        return segment_into_steps_double_newline(response_tokens, response_token_ids)
    else:
        return segment_into_steps_punctuation(response_tokens, response_token_ids)


def compute_step_entropy_scores(steps, token_entropies, token_top_ratio=0.2):
    """
    计算每个 step 的熵得分。

    方法：
    1. 找出全局 top token_top_ratio 高熵 token
    2. 对每个 step，累加落在该 step 内的高熵 token 的熵值

    Returns:
        step_scores: list[float]
        high_entropy_mask: np.ndarray (bool)
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
        score = entropy_in_step[mask_in_step].sum()
        step_scores.append(score)

    return step_scores, high_entropy_mask


def select_top_steps(steps, step_scores, step_top_ratio=0.1):
    """选择 top step_top_ratio 的高熵步骤，返回被选中的 step index 集合。"""
    n_steps = len(steps)
    if n_steps == 0:
        return set()
    top_k = max(1, int(np.ceil(n_steps * step_top_ratio)))
    sorted_indices = np.argsort(step_scores)[-top_k:]
    return set(sorted_indices.tolist())


# ──────────────────────────── Visualization ──────────────────────────

def print_separator(title="", width=100):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {C.BOLD}{title}{C.RESET} {'═' * pad}")
    else:
        print("═" * width)


def visualize_token_heatmap(response_tokens, token_entropies, high_entropy_mask):
    """打印 token 级别的熵热力图：高熵 token 加粗红色，低熵 token 暗色。"""
    max_ent = token_entropies.max() if len(token_entropies) > 0 else 1.0

    output = []
    for i, (tok, ent) in enumerate(zip(response_tokens, token_entropies)):
        display_tok = tok.replace('\n', '↵\n')
        if high_entropy_mask[i]:
            output.append(f"{C.BOLD}{C.RED}{display_tok}{C.RESET}")
        else:
            ratio = ent / (max_ent + 1e-6)
            if ratio < 0.33:
                output.append(f"{C.DIM}{display_tok}{C.RESET}")
            elif ratio < 0.66:
                output.append(f"{display_tok}")
            else:
                output.append(f"{C.YELLOW}{display_tok}{C.RESET}")

    print("".join(output))


def visualize_steps(steps, step_scores, high_entropy_mask, token_entropies,
                    step_top_ratio, token_top_ratio):
    """表格形式展示步骤，高亮被选中的步骤。"""
    selected = select_top_steps(steps, step_scores, step_top_ratio)

    header = (f"  {'Idx':>4}  {'Score':>7}  {'Tokens':>6}  "
              f"{'HE Tok':>6}  {'Sel':>3}  Text")
    print(f"{C.BOLD}{header}{C.RESET}")
    print("─" * 120)

    for j, (step, score) in enumerate(zip(steps, step_scores)):
        is_sel = j in selected
        n_he = sum(1 for idx in step['token_indices'] if high_entropy_mask[idx])
        text = step['text'].replace('\n', '↵ ').strip()

        mark = f"{C.GREEN}✅{C.RESET}" if is_sel else "  "
        color = C.RED if is_sel else ""
        reset = C.RESET if is_sel else ""

        print(f"  {color}{j:>4}  {score:>7.2f}  {len(step['token_indices']):>6}  "
              f"{n_he:>6}  {reset}{mark}  {color}{text}{reset}")


def visualize_one_result(result, token_top_ratio=0.2,
                          step_top_ratios=[0.1, 0.2, 0.3]):
    """对一个样本进行全方位可视化。"""
    steps = result['steps']
    tokens = result['response_tokens']
    entropies = result['token_entropies']

    if len(entropies) == 0:
        print("  (empty response, skipping)")
        return

    step_scores, high_entropy_mask = compute_step_entropy_scores(
        steps, entropies, token_top_ratio=token_top_ratio
    )

    # ---- Prompt ----
    print(f"\n{C.CYAN}Prompt:{C.RESET}")
    prompt_display = result['prompt']
    if len(prompt_display) > 500:
        prompt_display = prompt_display[:500] + "..."
    print(f"  {prompt_display}")

    # ---- Token heatmap ----
    print(f"\n{C.CYAN}Token-level entropy heatmap{C.RESET} "
          f"(total {len(tokens)} tokens, "
          f"{C.BOLD}{C.RED}bold red{C.RESET} = top {token_top_ratio*100:.0f}% high-entropy):")
    print()
    visualize_token_heatmap(tokens, entropies, high_entropy_mask)

    # ---- Step tables for different ratios ----
    for step_ratio in step_top_ratios:
        selected = select_top_steps(steps, step_scores, step_top_ratio=step_ratio)
        n_sel = len(selected)
        print_separator(
            f"Step view: top {step_ratio*100:.0f}% steps "
            f"({n_sel}/{len(steps)} selected) | "
            f"token_top_ratio={token_top_ratio*100:.0f}%"
        )
        visualize_steps(steps, step_scores, high_entropy_mask, entropies,
                        step_ratio, token_top_ratio)

    # ---- Stats ----
    print_separator("Statistics")
    top_k = max(1, int(np.ceil(len(entropies) * token_top_ratio)))
    threshold = np.sort(entropies)[-top_k]
    print(f"  Mean token entropy:  {entropies.mean():.4f}")
    print(f"  Median token entropy: {np.median(entropies):.4f}")
    print(f"  Max token entropy:   {entropies.max():.4f}")
    print(f"  Min token entropy:   {entropies.min():.4f}")
    print(f"  Top {token_top_ratio*100:.0f}% entropy threshold: {threshold:.4f}")
    print(f"  Step score mean:     {np.mean(step_scores):.4f}")
    print(f"  Step score std:      {np.std(step_scores):.4f}")
    print(f"  Total steps:         {len(steps)}")


def print_cross_ratio_comparison(results, step_top_ratio=0.2):
    """对比不同 token_top_ratio 对 step 选择的影响。"""
    print_separator(
        f"Cross comparison: token_top_ratio=10%/20%/30%, "
        f"step_top_ratio={step_top_ratio*100:.0f}%"
    )

    for i, r in enumerate(results):
        if len(r['token_entropies']) == 0:
            continue
        print(f"\n{'─'*80}")
        prompt_short = r['prompt'][:80].replace('\n', ' ')
        print(f"  {C.BOLD}Sample {i}{C.RESET}: {prompt_short}...")

        for t_ratio in [0.1, 0.2, 0.3]:
            scores, he_mask = compute_step_entropy_scores(
                r['steps'], r['token_entropies'], token_top_ratio=t_ratio
            )
            selected = select_top_steps(r['steps'], scores, step_top_ratio=step_top_ratio)

            print(f"\n    {C.YELLOW}token_top_ratio={t_ratio:.0%}{C.RESET}, "
                  f"selected {len(selected)}/{len(r['steps'])} steps:")
            for j in sorted(selected):
                s = r['steps'][j]
                text = s['text'].replace('\n', '↵ ').strip()
                print(f"      Step {j:>3} (score={scores[j]:>7.2f}): {text}")


def print_token_statistics(results, token_top_ratio=0.2):
    """统计高/低熵 token 的内容分布。"""
    print_separator(
        f"High vs Low entropy token content (token_top_ratio={token_top_ratio*100:.0f}%)"
    )

    all_high, all_low = [], []

    for r in results:
        entropies = r['token_entropies']
        tokens = r['response_tokens']
        if len(entropies) == 0:
            continue
        n = len(entropies)
        top_k = max(1, int(np.ceil(n * token_top_ratio)))
        top_indices = set(np.argsort(entropies)[-top_k:])

        for idx, tok in enumerate(tokens):
            tok_repr = repr(tok)
            if idx in top_indices:
                all_high.append(tok_repr)
            else:
                all_low.append(tok_repr)

    print(f"\n  Total high-entropy tokens: {len(all_high)}")
    print(f"  Total low-entropy tokens:  {len(all_low)}")

    print(f"\n  {C.RED}--- Top 40 most common HIGH entropy tokens ---{C.RESET}")
    for tok, cnt in Counter(all_high).most_common(40):
        print(f"    {tok:35s}  count={cnt}")

    print(f"\n  {C.BLUE}--- Top 40 most common LOW entropy tokens ---{C.RESET}")
    for tok, cnt in Counter(all_low).most_common(40):
        print(f"    {tok:35s}  count={cnt}")


# ──────────────────────────── Cache helpers ──────────────────────────

def save_cache(results, cache_path):
    """保存中间结果（token + entropy 数据）到 pickle 文件，供后续无 GPU 使用。"""
    # 保存前把 numpy array 转成 list 以便 pickle 兼容
    to_save = []
    for r in results:
        to_save.append({
            'prompt': r['prompt'],
            'response_text': r['response_text'],
            'response_tokens': r['response_tokens'],
            'response_token_ids': r['response_token_ids'],
            'token_entropies': r['token_entropies'].tolist()
                if hasattr(r['token_entropies'], 'tolist') else list(r['token_entropies']),
        })
    with open(cache_path, 'wb') as f:
        pickle.dump(to_save, f)
    print(f"  Cache saved to {cache_path}")


def load_cache(cache_path):
    """从 pickle 缓存加载中间结果，不需要 GPU。"""
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    results = []
    for r in cached:
        r['token_entropies'] = np.array(r['token_entropies'])
        results.append(r)
    print(f"  Loaded {len(results)} samples from cache: {cache_path}")
    return results


# ──────────────────────────── Main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step-level entropy visualization")
    parser.add_argument("--model", type=str,
                        default="/home/fuzhizhang.fzz/model/Qwen3-1.7B-Base",
                        help="Model path")
    parser.add_argument("--data", type=str,
                        default="/home/fuzhizhang.fzz/data/math__math_500.parquet",
                        help="Parquet data path")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of prompts to sample")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling prompts")
    parser.add_argument("--token_top_ratio", type=float, default=0.2,
                        help="Fraction of tokens considered high-entropy (default 0.2)")
    parser.add_argument("--step_top_ratios", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated step top ratios to visualize")
    parser.add_argument("--segment_method", type=str, default="punctuation",
                        choices=["punctuation", "double_newline"],
                        help="Step segmentation method: 'punctuation' (\\n + 句末标点) "
                             "or 'double_newline' (\\n\\n)")
    parser.add_argument("--save_cache", type=str, default=None,
                        help="Save intermediate token/entropy data to this pickle file")
    parser.add_argument("--load_cache", type=str, default=None,
                        help="Load intermediate data from cache (no GPU needed)")
    args = parser.parse_args()

    step_top_ratios = [float(x) for x in args.step_top_ratios.split(",")]

    if args.load_cache:
        # ---- CPU-only mode: load from cache ----
        print_separator("Loading from Cache (no GPU)")
        results = load_cache(args.load_cache)

        # Re-segment with the chosen method
        print_separator(f"Re-segmenting with method='{args.segment_method}'")
        for r in results:
            r['steps'] = segment_into_steps(
                r['response_tokens'], r['response_token_ids'],
                method=args.segment_method
            )
            print(f"  {len(r['response_tokens'])} tokens -> {len(r['steps'])} steps")
    else:
        # ---- GPU mode: generate and compute entropy ----
        if not _HAS_TORCH:
            raise RuntimeError("torch/transformers not installed. Use --load_cache for CPU mode.")

        device = "cuda:0"

        # ---- Load model ----
        print_separator("Loading Model")
        print(f"  Model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        print(f"  Loaded. Vocab size: {tokenizer.vocab_size}")

        # ---- Load data ----
        print_separator("Loading Data")
        df = pd.read_parquet(args.data)
        samples = df.sample(n=args.num_samples, random_state=args.seed)

        prompts = []
        for _, row in samples.iterrows():
            msgs = row["prompt"]
            if isinstance(msgs, (list, np.ndarray)):
                user_msg = str(msgs[0]["content"]) if len(msgs) > 0 else str(msgs)
            else:
                user_msg = str(msgs)
            prompts.append(user_msg)

        print(f"  Sampled {len(prompts)} prompts from {args.data}")

        # ---- Generate & compute entropy ----
        print_separator("Generating Responses")
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Generating...")
            resp_text, resp_tokens, resp_ids, entropies = generate_and_compute_entropy(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            r = {
                'prompt': prompt,
                'response_text': resp_text,
                'response_tokens': resp_tokens,
                'response_token_ids': resp_ids,
                'token_entropies': entropies,
            }
            r['steps'] = segment_into_steps(resp_tokens, resp_ids,
                                            method=args.segment_method)
            results.append(r)
            print(f"         {len(resp_tokens)} tokens, {len(r['steps'])} steps, "
                  f"avg entropy={entropies.mean():.3f}")

        # ---- Save cache if requested ----
        if args.save_cache:
            print_separator("Saving Cache")
            save_cache(results, args.save_cache)

    # ---- Per-sample visualization ----
    for i, r in enumerate(results):
        print_separator(f"📝 Sample {i}")
        visualize_one_result(r,
                             token_top_ratio=args.token_top_ratio,
                             step_top_ratios=step_top_ratios)

    # ---- Cross-ratio comparison ----
    print_cross_ratio_comparison(results, step_top_ratio=0.2)

    # ---- Token statistics ----
    print_token_statistics(results, token_top_ratio=args.token_top_ratio)

    print_separator("Done")


if __name__ == "__main__":
    main()
