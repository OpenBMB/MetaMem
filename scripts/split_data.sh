#!/bin/bash
INPUT_FILE="output/full_train_set.json" 
OUTPUT_DIR="data/folds_5_split"
LOGGING_FILE="logs/split_data.log"

N_FOLDS=5
TOTAL_SAMPLES=500
TEST_SIZE=100
VAL_SIZE=50
SEED=42

nohup python src/split_data.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n_folds "$N_FOLDS" \
    --total_samples "$TOTAL_SAMPLES" \
    --test_size "$TEST_SIZE" \
    --val_size "$VAL_SIZE" \
    --seed "$SEED" \
    >"$LOGGING_FILE" 2>&1 &