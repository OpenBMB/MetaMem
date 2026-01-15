#!/bin/bash
API_KEY="api-key"
BASE_URL="http://localhost:29001/v1"
MODEL_NAME="qwen3-30b"

LLMLINGUA_PATH="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
EMBEDDING_PATH="sentence-transformers/all-MiniLM-L6-v2"

INPUT_DATA="data/longmemeval_s_cleaned.json"
OUTPUT_DATA="output/memory_qwen3_30b.json"
QDRANT_DIR="qdrant/lightmem_qwen3_30b"
LOGGING_FILE="logs/construct_memory.log"

CUDA_VISIBLE_DEVICES=6 nohup python src/construct_memory.py \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --llm_model "$MODEL_NAME" \
    --llmlingua_path "$LLMLINGUA_PATH" \
    --embedding_path "$EMBEDDING_PATH" \
    --data_path "$INPUT_DATA" \
    --output_path "$OUTPUT_DATA" \
    --qdrant_dir "$QDRANT_DIR" \
    >"$LOGGING_FILE" 2>&1 &
