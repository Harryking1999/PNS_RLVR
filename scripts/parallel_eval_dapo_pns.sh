#!/usr/bin/env bash
# Parallel evaluation on 2 GPUs + PNS tests
# GPU 0: eval steps 200, 400, 600
# GPU 1: eval steps 737, 800, 850
# Then PNS on GPU 0 and GPU 1 in parallel
set -euo pipefail
cd /home/fuzhizhang.fzz/PNS_RLVR

PYTHON="/workspace/anaconda3/envs/verl_fzz/bin/python"
MERGE_DIR="/ssdwork/fuzhizhang/merged_models"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "======== Parallel Eval (2 GPUs) ========"
echo "  $(date)"

# ===== Phase 1: Parallel Eval =====
echo "===== Eval: GPU0=[step200,400,600] GPU1=[step737,800,850] ====="

CUDA_VISIBLE_DEVICES=0 ${PYTHON} tools/eval_benchmarks.py \
    --models \
        ${MERGE_DIR}/DAPO-PNS-step200 \
        ${MERGE_DIR}/DAPO-PNS-step400 \
        ${MERGE_DIR}/DAPO-PNS-step600 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output ${MERGE_DIR}/eval_results_gpu0.json \
    > ${MERGE_DIR}/eval_log_gpu0.txt 2>&1 &
PID0=$!
echo "  GPU0 started (PID: ${PID0})"

CUDA_VISIBLE_DEVICES=1 ${PYTHON} tools/eval_benchmarks.py \
    --models \
        ${MERGE_DIR}/DAPO-PNS-step737 \
        ${MERGE_DIR}/DAPO-PNS-step800 \
        ${MERGE_DIR}/DAPO-PNS-step850 \
    --benchmarks aime24 aime25 math500 math500_noop \
    --max_tokens 16384 \
    --output ${MERGE_DIR}/eval_results_gpu1.json \
    > ${MERGE_DIR}/eval_log_gpu1.txt 2>&1 &
PID1=$!
echo "  GPU1 started (PID: ${PID1})"

echo "  Waiting for both evals to finish..."
wait ${PID0}
echo "  GPU0 eval done!"
wait ${PID1}
echo "  GPU1 eval done!"

# Merge the two result files
echo "  Merging eval results..."
${PYTHON} -c "
import json
r0 = json.load(open('${MERGE_DIR}/eval_results_gpu0.json'))
r1 = json.load(open('${MERGE_DIR}/eval_results_gpu1.json'))
r0.update(r1)
with open('${MERGE_DIR}/eval_results.json', 'w') as f:
    json.dump(r0, f, indent=2, ensure_ascii=False)
print(f'  Merged {len(r0)} models into eval_results.json')
"

# ===== Phase 2: PNS Estimation (parallel on 2 GPUs) =====
echo ""
echo "===== PNS Estimation: GPU0=[step200,600] GPU1=[step850] ====="

CUDA_VISIBLE_DEVICES=0 ${PYTHON} PNS_test/compute_pns.py \
    --model ${MERGE_DIR}/DAPO-PNS-step200 \
    --benchmark math500 --num_problems 50 --num_rollouts 5 \
    --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --resume \
    > ${MERGE_DIR}/pns_log_step200.txt 2>&1 &
PNS_PID0=$!

CUDA_VISIBLE_DEVICES=1 ${PYTHON} PNS_test/compute_pns.py \
    --model ${MERGE_DIR}/DAPO-PNS-step850 \
    --benchmark math500 --num_problems 50 --num_rollouts 5 \
    --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --resume \
    > ${MERGE_DIR}/pns_log_step850.txt 2>&1 &
PNS_PID1=$!

echo "  PNS GPU0=step200 (PID: ${PNS_PID0}), GPU1=step850 (PID: ${PNS_PID1})"
echo "  Waiting..."
wait ${PNS_PID0}
echo "  PNS step200 done!"
wait ${PNS_PID1}
echo "  PNS step850 done!"

# Third PNS run (step600) on GPU 0
echo "  PNS: step600 on GPU0..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} PNS_test/compute_pns.py \
    --model ${MERGE_DIR}/DAPO-PNS-step600 \
    --benchmark math500 --num_problems 50 --num_rollouts 5 \
    --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 --resume \
    > ${MERGE_DIR}/pns_log_step600.txt 2>&1
echo "  PNS step600 done!"

# Compare PNS results
echo ""
echo "  Running PNS comparison..."
${PYTHON} PNS_test/compare_pns_results.py

echo ""
echo "======== All Done! $(date) ========"
echo "  Eval results: ${MERGE_DIR}/eval_results.json"
echo "  PNS results:  /ssdwork/fuzhizhang/pns_results/"
echo "  Eval logs:    ${MERGE_DIR}/eval_log_gpu{0,1}.txt"
echo "  PNS logs:     ${MERGE_DIR}/pns_log_step{200,600,850}.txt"
