#!/bin/bash
EXPERIMENT_DIR="data/memory/train/fold1_run"
META_MEMORIES="$EXPERIMENT_DIR/step_x/meta_memories.json" 
DATASET="data/folds_5_split/fold_1_test.json"
OUTPUT="output/infer_fold1_test_step1.jsonl"
LOGGING_FILE="logs/infer_metamem.log"

API_KEY="xxx"
BASE_URL="http://localhost:29001/v1"
MODEL_NAME="qwen3-30b"
JUDGE_API_KEY="xxx"
JUDGE_BASE_URL="http://localhost:29002/v1"
JUDGE_MODEL_NAME="qwen3-235b"

ROLLOUT_CONCURRENCY=16
TEMPERATURE=0.0
MAX_TOKENS=4096

nohup python src/infer_metamem.py \
    --dataset "$DATASET" \
    --meta_memories "$META_MEMORIES" \
    --output "$OUTPUT" \
    --rollout_concurrency "$ROLLOUT_CONCURRENCY" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --judge_api_key "$JUDGE_API_KEY" \
    --judge_base_url "$JUDGE_BASE_URL" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    >"$LOGGING_FILE" 2>&1 &
