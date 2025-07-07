#!/bin/bash

# Check argument count
if [ $# -lt 7 ]; then
    echo "Usage: $0 <cuda_devices> <dataset> <model> <backbone> <quant> <exp_tag> <affine_tag> [additional_args]"
    echo "Example: $0 0,1 COCO sd3 dit intw8a8 baseline blockwise --additional-arg value"
    exit 1
fi

# Parse arguments
DATASET=$2
MODEL=$3
BACKBONE=$4
QUANT=$5
EXP_TAG=$6
AFFINE_TAG=$7

# Compose log filename
LOG_NAME="${DATASET,,}-${MODEL,,}-${BACKBONE,,}-${QUANT,,}-${EXP_TAG,,}-${AFFINE_TAG,,}.log"

# If log file exists, do not run
if [ -f "$LOG_NAME" ]; then
    echo "Log file $LOG_NAME already exists. Aborting."
    exit 1
fi

# Run command
nohup env \
    CUDA_VISIBLE_DEVICES="$1" \
    python3 -m benchmark.affiner_expr \
    -d "$DATASET" \
    -m "$MODEL" \
    -l "$BACKBONE" \
    -q "$QUANT" \
    -e "$EXP_TAG" \
    -a "$AFFINE_TAG" \
    "${@:8}" \
    > "$LOG_NAME" 2>&1 &

echo "Launched with log: $LOG_NAME"
