#!/usr/bin/env python3
"""
Evaluate merged HF checkpoints on AIME24, AIME25, MATH500, MATH500_Noop.
Uses vLLM for fast batched generation and verl's naive_dapo scorer.

Metrics:
  Greedy (temp=0, n=1):   Acc@1, Len@1
  Sampling (temp=1.0, n=16): Acc@16 (avg accuracy), Pass@16 (any correct), Len@16 (avg len)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/eval_benchmarks.py \
        --models /ssdwork/fuzhizhang/merged_models/DAPO-step400 \
        --benchmarks aime24 aime25 math500 \
        --max_tokens 16384 \
        --output /ssdwork/fuzhizhang/eval_results.json
"""

import argparse
import gc
import json
import multiprocessing
import os
import queue as queue_module
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add project root to path0
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from verl.utils.reward_score.naive_dapo import compute_score


# ─────────────────── Robust timeout wrapper ────────────────────

def _compute_score_worker(q, text, answer):
    """Worker function for multiprocessing-based timeout."""
    try:
        result = compute_score(text, answer)
        q.put(result)
    except Exception:
        q.put({"score": 0.0, "acc": False})


def compute_score_safe(text: str, answer: str, timeout_seconds: int = 60) -> dict:
    """Wrapper around compute_score with process-level timeout.
    
    sympy.simplify() can hang forever in C code where signal-based timeouts
    (SIGALRM) cannot interrupt. This uses multiprocessing to forcefully kill
    the scoring process if it exceeds the timeout.
    """
    q = multiprocessing.Queue(maxsize=1)
    p = multiprocessing.Process(target=_compute_score_worker, args=(q, text, answer))
    p.start()
    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join()
        print(f"  [WARN] compute_score timed out after {timeout_seconds}s, marking as incorrect")
        return {"score": 0.0, "acc": False}

    try:
        return q.get(timeout=0.1)
    except queue_module.Empty:
        return {"score": 0.0, "acc": False}
    finally:
        q.close()


# ─────────────────── Data loading ────────────────────

def load_aime24():
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


def load_aime25():
    path = "/home/fuzhizhang.fzz/data/aime25/test.jsonl"
    problems = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "id": f"aime25_{data['id']}",
                "problem": data["problem"],
                "answer": str(data["answer"]),
            })
    return problems


def load_math500():
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


def load_math500_noop():
    path = "/home/fuzhizhang.fzz/CoT_Causal_Analysis/data/MATH500_noop/test.jsonl"
    problems = []
    with open(path) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            problems.append({
                "id": f"math500_noop_{idx}",
                "problem": data["problem"],
                "answer": str(data["answer"]),
            })
    return problems


BENCHMARK_LOADERS = {
    "aime24": load_aime24,
    "aime25": load_aime25,
    "math500": load_math500,
    "math500_noop": load_math500_noop,
}


# ─────────────────── Prompt building ────────────────────

SUFFIX = "\nPlease output the final answer within \\boxed{}."


def build_prompt(problem_text: str, tokenizer) -> str:
    if "\\boxed{}" in problem_text and "Please output" in problem_text:
        user_content = problem_text
    else:
        user_content = problem_text + SUFFIX
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ─────────────────── Scoring ────────────────────

def score_greedy(problems, outputs):
    """Score greedy outputs → Acc@1, Len@1."""
    correct = 0
    total_len = 0
    for i, (prob, output) in enumerate(zip(problems, outputs)):
        resp = output.outputs[0]
        result = compute_score_safe(resp.text, prob["answer"])
        if bool(result["acc"]):
            correct += 1
        total_len += len(resp.token_ids)
        if (i + 1) % 100 == 0:
            print(f"    scored {i+1}/{len(problems)}")
    acc = correct / len(problems) * 100
    avg_len = total_len / len(problems)
    return {"acc_at_1": acc, "len_at_1": avg_len,
            "correct": correct, "total": len(problems)}


def score_at_n(problems, outputs, n):
    """Score sampling outputs → Acc@16, Pass@16, Len@16."""
    sum_acc = 0.0
    pass_count = 0
    total_len = 0
    total_samples = 0
    for i, (prob, output) in enumerate(zip(problems, outputs)):
        n_correct = 0
        for out in output.outputs:
            result = compute_score_safe(out.text, prob["answer"])
            if bool(result["acc"]):
                n_correct += 1
            total_len += len(out.token_ids)
            total_samples += 1
        # Per-problem accuracy
        sum_acc += n_correct / n
        # Pass@16: any correct
        if n_correct > 0:
            pass_count += 1
        if (i + 1) % 50 == 0:
            print(f"    scored {i+1}/{len(problems)} problems ({total_samples} samples)")
    acc_at_n = sum_acc / len(problems) * 100
    pass_at_n = pass_count / len(problems) * 100
    avg_len = total_len / max(total_samples, 1)
    return {"acc_at_16": acc_at_n, "pass_at_16": pass_at_n, "len_at_16": avg_len,
            "pass_count": pass_count, "total": len(problems)}


# ─────────────────── IO helpers ────────────────────

def load_existing_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(all_results, output_path):
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def print_table(all_results, benchmarks):
    """Print a formatted results table."""
    # Columns: for each benchmark → Acc@1, Len@1, Acc@16, Pass@16, Len@16
    metrics = ["acc_at_1", "len_at_1", "acc_at_16", "pass_at_16", "len_at_16"]
    metric_labels = {"acc_at_1": "Acc@1", "len_at_1": "Len@1",
                     "acc_at_16": "Acc@16", "pass_at_16": "Pass@16", "len_at_16": "Len@16"}

    # Header
    header = f"{'Model':<40}"
    for b in benchmarks:
        for m in metrics:
            label = f"{b}/{metric_labels[m]}"
            header += f"{label:>16}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print("-" * len(header))

    for model_name in sorted(all_results.keys()):
        row = f"{model_name:<40}"
        for b in benchmarks:
            for m in metrics:
                val = all_results.get(model_name, {}).get(b, {}).get(m)
                if val is not None:
                    if "len" in m:
                        row += f"{val:>15.0f}"
                    else:
                        row += f"{val:>15.1f}%"
                else:
                    row += f"{'—':>16}"
        print(row)
    print(sep)


# ─────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on math benchmarks")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "math500"],
                        choices=list(BENCHMARK_LOADERS.keys()))
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--output", type=str, default="/ssdwork/fuzhizhang/eval_results.json")
    args = parser.parse_args()

    all_results = load_existing_results(args.output)

    for model_path in args.models:
        model_name = Path(model_path).name
        if model_name not in all_results:
            all_results[model_name] = {}

        # Check what's already done for this model
        done_benchmarks = set()
        for b in args.benchmarks:
            entry = all_results[model_name].get(b, {})
            # Complete if has all 5 metrics
            if all(k in entry for k in ["acc_at_1", "len_at_1", "acc_at_16", "pass_at_16", "len_at_16"]):
                done_benchmarks.add(b)

        remaining = [b for b in args.benchmarks if b not in done_benchmarks]
        if not remaining:
            print(f"\nSkipping {model_name} — all benchmarks done")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"  Remaining benchmarks: {remaining}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=args.max_tokens + 2048,
            dtype="bfloat16",
        )

        sp_greedy = SamplingParams(temperature=0, max_tokens=args.max_tokens, n=1)
        sp_sample = SamplingParams(temperature=1.0, max_tokens=args.max_tokens, n=16)

        for bench_name in remaining:
            print(f"\n--- {bench_name} ---")
            problems = BENCHMARK_LOADERS[bench_name]()
            prompts = [build_prompt(p["problem"], tokenizer) for p in problems]
            print(f"  {len(problems)} problems")

            # === Greedy (Acc@1, Len@1) ===
            entry = all_results[model_name].get(bench_name, {})
            if "acc_at_1" not in entry:
                print(f"  [greedy] generating...")
                t0 = time.time()
                outputs = llm.generate(prompts, sp_greedy)
                print(f"  [greedy] generated in {time.time()-t0:.1f}s, scoring...")
                result = score_greedy(problems, outputs)
                del outputs; gc.collect()
                entry.update(result)
                print(f"  [greedy] Acc@1={result['acc_at_1']:.1f}%, Len@1={result['len_at_1']:.0f}")
                all_results[model_name][bench_name] = entry
                save_results(all_results, args.output)
            else:
                print(f"  [greedy] already done, Acc@1={entry['acc_at_1']:.1f}%")

            # === Sampling (Acc@16, Pass@16, Len@16) ===
            if "acc_at_16" not in entry:
                print(f"  [@16] generating (temp=1.0, n=16)...")
                t0 = time.time()
                outputs = llm.generate(prompts, sp_sample)
                print(f"  [@16] generated in {time.time()-t0:.1f}s, scoring...")
                result = score_at_n(problems, outputs, 16)
                del outputs; gc.collect()
                entry.update(result)
                print(f"  [@16] Acc@16={result['acc_at_16']:.1f}%, Pass@16={result['pass_at_16']:.1f}%, Len@16={result['len_at_16']:.0f}")
                all_results[model_name][bench_name] = entry
                save_results(all_results, args.output)
            else:
                print(f"  [@16] already done, Acc@16={entry['acc_at_16']:.1f}%")

        del llm; gc.collect(); torch.cuda.empty_cache()

    # Final summary
    save_results(all_results, args.output)
    print_table(all_results, args.benchmarks)


if __name__ == "__main__":
    main()
