#!/bin/bash
# Wait for benchmark eval to finish, then run PNS forward + reverse for 3 dry-run models
# GPU 0: step200 + step600;  GPU 1: step400

PY=/workspace/anaconda3/envs/verl_fzz/bin/python
MERGE_DIR=/home/fuzhizhang.fzz/model/merge_models
PNS_OUT=/home/fuzhizhang.fzz/model/merge_models/pns_results

EVAL_PID_GPU0=$1
EVAL_PID_GPU1=$2

echo "[$(date)] Waiting for eval processes to finish..."
echo "  GPU0 eval PID: $EVAL_PID_GPU0"
echo "  GPU1 eval PID: $EVAL_PID_GPU1"

# Wait for both eval processes
if [ -n "$EVAL_PID_GPU0" ]; then
    while kill -0 $EVAL_PID_GPU0 2>/dev/null; do sleep 30; done
    echo "[$(date)] GPU0 eval finished"
fi
if [ -n "$EVAL_PID_GPU1" ]; then
    while kill -0 $EVAL_PID_GPU1 2>/dev/null; do sleep 30; done
    echo "[$(date)] GPU1 eval finished"
fi

echo "[$(date)] All evals done. Starting PNS estimation..."

# ─── GPU 0: step200 forward + reverse, then step600 forward + reverse ───
(
    for STEP in 200 600; do
        MODEL=${MERGE_DIR}/DAPO-PNS-dryrun-step${STEP}
        OUT=${PNS_OUT}/DAPO-PNS-dryrun-step${STEP}
        echo "[$(date)] GPU0: PNS forward for step${STEP}"
        CUDA_VISIBLE_DEVICES=0 $PY PNS_test/compute_pns.py \
            --model $MODEL --benchmark math500 \
            --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
            --ablation_mode counterfactual \
            --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 \
            --test_all_steps --resume \
            --output_dir $OUT
        
        echo "[$(date)] GPU0: PNS reverse for step${STEP}"
        CUDA_VISIBLE_DEVICES=0 $PY PNS_test/compute_pns.py \
            --model $MODEL --benchmark math500 \
            --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
            --ablation_mode counterfactual \
            --test_all_steps --reverse --resume \
            --output_dir $OUT
    done
    echo "[$(date)] GPU0: All PNS done"
) &
PNS_PID_GPU0=$!

# ─── GPU 1: step400 forward + reverse ───
(
    MODEL=${MERGE_DIR}/DAPO-PNS-dryrun-step400
    OUT=${PNS_OUT}/DAPO-PNS-dryrun-step400
    echo "[$(date)] GPU1: PNS forward for step400"
    CUDA_VISIBLE_DEVICES=1 $PY PNS_test/compute_pns.py \
        --model $MODEL --benchmark math500 \
        --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
        --ablation_mode counterfactual \
        --entropy_top_ratio 0.2 --step_entropy_top_ratio 0.2 \
        --test_all_steps --resume \
        --output_dir $OUT
    
    echo "[$(date)] GPU1: PNS reverse for step400"
    CUDA_VISIBLE_DEVICES=1 $PY PNS_test/compute_pns.py \
        --model $MODEL --benchmark math500 \
        --num_problems 50 --num_rollouts 5 --max_tokens 16384 \
        --ablation_mode counterfactual \
        --test_all_steps --reverse --resume \
        --output_dir $OUT
    
    echo "[$(date)] GPU1: All PNS done"
) &
PNS_PID_GPU1=$!

echo "[$(date)] PNS launched: GPU0 PID=$PNS_PID_GPU0, GPU1 PID=$PNS_PID_GPU1"
wait $PNS_PID_GPU0 $PNS_PID_GPU1
echo "[$(date)] All PNS estimation complete!"
