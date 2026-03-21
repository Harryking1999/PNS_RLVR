#!/usr/bin/env python3
"""
PNS Results Comparison Tool

汇总多个模型的 PNS 估算结果，输出对比表格。

用法:
    python PNS_test/compare_pns_results.py
    python PNS_test/compare_pns_results.py --results_dir /home/fuzhizhang.fzz/model/merge_models/pns_results
"""

import argparse
import json
import os
from pathlib import Path


def load_analysis(results_dir, model_name):
    """加载某个模型的 pns_analysis.json"""
    path = os.path.join(results_dir, model_name, "pns_analysis.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare PNS results across models")
    parser.add_argument("--results_dir", type=str,
                        default="/home/fuzhizhang.fzz/model/merge_models/pns_results",
                        help="Directory containing per-model PNS results")
    args = parser.parse_args()

    # 定义模型分组
    model_groups = {
        "Qwen2.5-1.5B 系列 (Base → SFT → Distill → Distill+RL)": [
            "Qwen2.5-1.5B-Base",
            "Qwen2.5-1.5B-Instruct",
            "DeepSeek-R1-Distill-Qwen-1.5B",
            "DeepScaleR-1.5B-Preview",
        ],
        "Qwen3-1.7B 系列 (Base → RL)": [
            "Qwen3-1.7B-Base",
            "Qwen3-1.7B",
        ],
    }

    # 也尝试加载目录中所有其他结果
    all_dirs = []
    if os.path.isdir(args.results_dir):
        all_dirs = sorted([
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        ])

    known_models = set()
    for models in model_groups.values():
        known_models.update(models)
    extra_models = [d for d in all_dirs if d not in known_models]
    if extra_models:
        model_groups["其他模型"] = extra_models

    # 收集所有结果
    all_results = {}
    for group_name, models in model_groups.items():
        for model_name in models:
            analysis = load_analysis(args.results_dir, model_name)
            if analysis:
                all_results[model_name] = analysis

    if not all_results:
        print(f"未找到任何 PNS 结果，请先运行 PNS 估算脚本。")
        print(f"结果目录: {args.results_dir}")
        return

    # ──────────────────────────────────────────────────────
    # 打印对比表格
    # ──────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("  PNS 结果对比")
    print("=" * 120)

    for group_name, models in model_groups.items():
        group_results = {m: all_results[m] for m in models if m in all_results}
        if not group_results:
            continue

        print(f"\n{'─'*120}")
        print(f"  {group_name}")
        print(f"{'─'*120}")

        # Header
        header = f"{'Model':<38} {'Solved':>7} {'Steps':>6} "
        header += f"{'PNS_mean':>9} {'PNS_med':>8} {'PNS_std':>8} "
        header += f"{'PNS≥0.6':>8} {'PNS≤0.2':>8} "
        header += f"{'Ent↔PNS':>8}"

        # If test_all_steps, add HE vs LE comparison
        has_he_le = any(
            "high_vs_low_entropy" in all_results.get(m, {})
            for m in models if m in all_results
        )
        if has_he_le:
            header += f" {'HE_PNS':>7} {'LE_PNS':>7} {'Δ(H-L)':>7}"

        print(header)
        print("─" * len(header))

        for model_name in models:
            if model_name not in all_results:
                print(f"  {model_name:<36}  (未完成)")
                continue

            a = all_results[model_name]
            s = a.get("summary", {})
            ps = a.get("pns_stats", {})
            dist = a.get("pns_distribution", {})

            n_solved = s.get("num_problems_with_correct", 0)
            n_steps = s.get("num_steps_tested", 0)
            pns_mean = ps.get("mean", 0)
            pns_med = ps.get("median", 0)
            pns_std = ps.get("std", 0)

            # Count PNS >= 0.6 and <= 0.2
            high_pns = sum(v for k, v in dist.items()
                          if any(x in k for x in ["0.6", "0.8", "=1.0"]))
            low_pns = sum(v for k, v in dist.items()
                         if any(x in k for x in ["=0.0", "≤0.2"]))

            high_pns_pct = high_pns / max(n_steps, 1) * 100
            low_pns_pct = low_pns / max(n_steps, 1) * 100

            corr = a.get("entropy_pns_correlation", None)
            corr_str = f"{corr:>8.3f}" if corr is not None else f"{'N/A':>8}"

            row = (f"  {model_name:<36} {n_solved:>7} {n_steps:>6} "
                   f"{pns_mean:>9.4f} {pns_med:>8.4f} {pns_std:>8.4f} "
                   f"{high_pns_pct:>7.1f}% {low_pns_pct:>7.1f}% "
                   f"{corr_str}")

            if has_he_le:
                hv = a.get("high_vs_low_entropy", {})
                if hv:
                    he_pns = hv.get("high_entropy_mean_pns", 0)
                    le_pns = hv.get("low_entropy_mean_pns", 0)
                    delta = hv.get("delta_mean", 0)
                    row += f" {he_pns:>7.3f} {le_pns:>7.3f} {delta:>+7.3f}"
                else:
                    row += f" {'N/A':>7} {'N/A':>7} {'N/A':>7}"

            print(row)

    # ──────────────────────────────────────────────────────
    # Position analysis comparison
    # ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PNS by Step Position (Early / Middle / Late)")
    print(f"{'='*80}")

    header2 = f"{'Model':<38}"
    for pos in ["early", "middle", "late"]:
        header2 += f" {pos+'_mean':>11} {pos+'_n':>6}"
    print(header2)
    print("─" * len(header2))

    for model_name, a in all_results.items():
        pos_stats = a.get("position_stats", {})
        if not pos_stats:
            continue
        row = f"  {model_name:<36}"
        for pos in ["early", "middle", "late"]:
            ps = pos_stats.get(pos, {})
            if ps:
                row += f" {ps['mean_pns']:>11.4f} {ps['count']:>6}"
            else:
                row += f" {'N/A':>11} {'N/A':>6}"
        print(row)

    # ──────────────────────────────────────────────────────
    # Key takeaways
    # ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Key Takeaways")
    print(f"{'='*80}")

    if len(all_results) >= 2:
        sorted_by_pns = sorted(all_results.items(),
                               key=lambda x: x[1].get("pns_stats", {}).get("mean", 0),
                               reverse=True)
        print(f"\n  模型按 mean PNS 排序 (高→低, 高 PNS = 步骤更不可或缺):")
        for i, (name, a) in enumerate(sorted_by_pns):
            pns = a.get("pns_stats", {}).get("mean", 0)
            print(f"    {i+1}. {name:<36} mean_PNS = {pns:.4f}")

        # Check if RL models have different PNS patterns
        print(f"\n  分析:")
        print(f"  • 如果 RL/Distill 模型的 mean PNS 高于 Base 模型 →")
        print(f"    说明训练后模型的推理步骤更加不可或缺 (更紧凑、更关键)")
        print(f"  • 如果 RL/Distill 模型的 mean PNS 低于 Base 模型 →")
        print(f"    说明训练后模型有更多冗余步骤 (可能更 verbose)")
        print(f"  • Entropy↔PNS 相关性越高 → entropy 越能代替 PNS")
        print(f"    相关性越低 → PNS 提供了 entropy 无法捕获的新信号")

    print()


if __name__ == "__main__":
    main()
