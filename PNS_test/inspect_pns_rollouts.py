"""
Inspect PNS rollouts: for high-PNS steps, show the counterfactual prompt
and the actual generated responses to judge quality.
"""
import json
import sys
import os
import torch

torch.multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SUFFIX = "\nPlease output the final answer within \\boxed{}."

COUNTERFACTUAL_INSTRUCTION_TEMPLATE = (
    "You have been working on this problem and have made progress as shown above. "
    "However, for the next step, you must NOT use the following reasoning:\n"
    "\"{FORBIDDEN}\"\n\n"
    "Please find an alternative approach to continue solving the problem. "
    "Do NOT repeat or rephrase the forbidden step. "
    "Output the final answer within \\boxed{}."
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of high-PNS examples to inspect")
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Short gen for quick inspection")
    args = parser.parse_args()

    # Load data
    phase2 = json.load(open(os.path.join(args.results_dir, "phase2_entropy_steps.json")))
    phase3 = json.load(open(os.path.join(args.results_dir, "phase3_ablation_rollouts.json")))

    # Filter meaningful high-PNS steps
    high_pns = [d for d in phase3 if d["pns"] >= 0.8 and len(d["step_text_preview"].strip()) > 15]
    # Sort by PNS desc, then diverse by problem
    high_pns.sort(key=lambda x: (-x["pns"], x["problem_id"]))

    # Pick diverse examples (different problems)
    selected = []
    seen_problems = set()
    for d in high_pns:
        pid = d["problem_id"]
        if pid in seen_problems:
            continue
        seen_problems.add(pid)
        selected.append(d)
        if len(selected) >= args.num_examples:
            break

    print(f"Selected {len(selected)} high-PNS steps to inspect")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build counterfactual prompts
    prompts = []
    for sel in selected:
        # Find the corresponding phase2 item
        item = None
        for it in phase2:
            if it["id"] == sel["problem_id"]:
                item = it
                break
        if item is None:
            print(f"  WARNING: could not find phase2 data for {sel['problem_id']}")
            continue

        response_tokens = item["response_tokens"]
        # Find the step
        step = None
        for s in item["steps"]:
            if s["step_idx"] == sel["step_idx"]:
                step = s
                break
        if step is None:
            print(f"  WARNING: step {sel['step_idx']} not found")
            continue

        # Build counterfactual prefix
        prefix_text = ''.join(response_tokens[:step['start']])
        forbidden_text = step['text'].strip()

        if "\\boxed{}" in item["problem"] and "Please output" in item["problem"]:
            user_content = item["problem"]
        else:
            user_content = item["problem"] + SUFFIX

        cf_msg = COUNTERFACTUAL_INSTRUCTION_TEMPLATE.replace("{FORBIDDEN}", forbidden_text)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prefix_text},
            {"role": "user", "content": cf_msg},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append((sel, item, step, prefix_text, forbidden_text, prompt))

    if not prompts:
        print("No prompts to generate!")
        return

    # Load vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_tokens + 4096,
        dtype="bfloat16",
    )

    sp = SamplingParams(
        temperature=1.0,
        max_tokens=args.max_tokens,
        n=args.num_rollouts,
    )

    # Generate
    prompt_texts = [p[5] for p in prompts]
    outputs = llm.generate(prompt_texts, sp)

    # Display results
    for (sel, item, step, prefix_text, forbidden_text, _), output in zip(prompts, outputs):
        print("\n" + "=" * 80)
        print(f"Problem: {sel['problem_id']} | Step: {sel['step_idx']} | PNS: {sel['pns']}")
        print(f"Position: {sel['position']} | High-Entropy: {sel['is_high_entropy']}")
        print("-" * 80)

        # Show the problem (truncated)
        prob_text = item["problem"]
        if len(prob_text) > 300:
            prob_text = prob_text[:300] + "..."
        print(f"PROBLEM: {prob_text}")
        print("-" * 80)

        # Show prefix (what the model had already reasoned)
        if len(prefix_text) > 500:
            print(f"PROGRESS SO FAR (first 500 chars): {prefix_text[:500]}...")
        else:
            print(f"PROGRESS SO FAR: {prefix_text}")
        print("-" * 80)

        # Show the forbidden step
        print(f"FORBIDDEN STEP: {forbidden_text}")
        print(f"CORRECT ANSWER: {item['answer']}")
        print("-" * 80)

        # Show each rollout response
        from verl.utils.reward_score.naive_dapo import compute_score
        for i, out in enumerate(output.outputs):
            resp = out.text
            result = compute_score(resp, item["answer"])
            correct = bool(result["acc"])
            status = "✅ CORRECT" if correct else "❌ WRONG"

            # Show first 600 chars of response
            resp_display = resp[:600]
            if len(resp) > 600:
                resp_display += f"... [total {len(resp)} chars]"

            print(f"\n  Rollout {i+1} [{status}] (len={len(out.token_ids)} tokens):")
            print(f"  {resp_display}")

        print()


if __name__ == "__main__":
    main()
