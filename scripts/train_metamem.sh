#!/bin/bash
EXPERIMENT_NAME="fold1_run"
DATASET="data/folds_5_split/fold_1_train.json"
LOGGING_FILE="logs/train_metamem.log"

API_KEY="xxx"
BASE_URL="http://localhost:29001/v1"
MODEL_NAME="qwen3-30b"
JUDGE_BASE_URL="http://localhost:29002/v1"
JUDGE_MODEL_NAME="qwen3-235b"

EPOCHS=5
BATCHSIZE=50
NUM_SAMPLES=5
ROLLOUT_CONCURRENCY=16
TEMPERATURE=0.7
MAX_TOKENS=4096

nohup python src/train_metamem.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batchsize "$BATCHSIZE" \
    --num_samples "$NUM_SAMPLES" \
    --rollout_concurrency "$ROLLOUT_CONCURRENCY" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --judge_base_url "$JUDGE_BASE_URL" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    >"$LOGGING_FILE" 2>&1 &
