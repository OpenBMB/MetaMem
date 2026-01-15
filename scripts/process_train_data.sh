#!/bin/bash
RAW_DATA="data/longmemeval_s_cleaned.json"
PROCESSED_DATA="output/memory_qwen3_30b.json"

OUTPUT_FILE="output/full_train_set.json"
LOGGING_FILE="logs/process_train_data.log"

nohup python src/process_train_data.py \
    --raw_data_path "$RAW_DATA" \
    --processed_data_path "$PROCESSED_DATA" \
    --output_path "$OUTPUT_FILE" \
    >"$LOGGING_FILE" 2>&1 &