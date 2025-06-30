#!/bin/bash

# Check argument count
if [ $# -lt 6 ]; then
    echo "Usage: $0 <cuda_devices> <dataset> <model> <backbone> <quant> <exp_tag>"
    echo "Example: $0 0 COCO sd3 dit intw8a8 segquant-gptq"
    exit 1
fi

# Parse arguments
export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
MODEL=$3
BACKBONE=$4
QUANT=$5
EXP_TAG=$6

# Compose log filename
LOG_NAME="${DATASET,,}-${MODEL,,}-${BACKBONE,,}-${QUANT,,}-${EXP_TAG,,}.log"

# Run command
nohup python3 -m benchmark.main_expr \
    -d "$DATASET" \
    -m "$MODEL" \
    -l "$BACKBONE" \
    -q "$QUANT" \
    -e "$EXP_TAG" \
    --config-dir config \
    -C config/calibrate_config.json \
    "${@:7}" \
    > "$LOG_NAME" 2>&1 &

echo "Launched with log: $LOG_NAME"
