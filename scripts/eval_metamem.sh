#!/bin/bash
EXPERIMENT_DIR="data/memory/train/fold1_run"
DATASET="data/folds_5_split/fold_1_val.json"
OUTPUT_DIR="eval_results"
OUTPUT_FILE="$OUTPUT_DIR/eval_all_steps.json"
LOGGING_FILE="logs/eval_metamem.log"

API_KEY="xxx"
BASE_URL="http://localhost:29001/v1"
MODEL_NAME="qwen3-30b"
JUDGE_API_KEY="xxx"
JUDGE_BASE_URL="http://localhost:29002/v1"
JUDGE_MODEL_NAME="qwen3-235b"

ROLLOUT_CONCURRENCY=16
MAX_TOKENS=4096
START_STEP=1

nohup python src/eval_metamem.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --rollout_concurrency "$ROLLOUT_CONCURRENCY" \
    --max_tokens "$MAX_TOKENS" \
    --start_step "$START_STEP" \
    --no_baseline \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --judge_api_key "$JUDGE_API_KEY" \
    --judge_base_url "$JUDGE_BASE_URL" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    >"$LOGGING_FILE" 2>&1 &
