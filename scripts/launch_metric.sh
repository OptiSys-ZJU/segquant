#!/bin/bash

# Check argument count
if [ $# -lt 6 ]; then
    echo "Usage: $0 <cuda_devices> <dataset> <model> <backbone> <quant> <exp_tag> [affine]"
    echo "Example: $0 0,1 COCO sd3 dit intw8a8 segquant-gptq ptqd"
    exit 1
fi

# Parse required arguments
CUDA_DEVICES=$1
DATASET=$2
MODEL=$3
BACKBONE=$4
QUANT=$5
EXP_TAG=$6

# Optional attribute argument (7th)
if [ $# -ge 7 ]; then
    AFF=$7
    AFF_ARG="-a $AFF"
    LOG_NAME="metric-${DATASET,,}-${MODEL,,}-${BACKBONE,,}-${QUANT,,}-${EXP_TAG,,}-${AFF,,}.log"
    EXTRA_ARGS="${@:8}"  # remaining args after attr
else
    AFF_ARG=""
    LOG_NAME="metric-${DATASET,,}-${MODEL,,}-${BACKBONE,,}-${QUANT,,}-${EXP_TAG,,}.log"
    EXTRA_ARGS="${@:7}"  # remaining args if no attr
fi

# Check for existing log
if [ -f "$LOG_NAME" ]; then
    echo "Log file $LOG_NAME already exists. Aborting."
    exit 1
fi

# Run command
nohup env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    python3 -m benchmark.metric \
    -d "$DATASET" \
    -m "$MODEL" \
    -l "$BACKBONE" \
    -q "$QUANT" \
    -e "$EXP_TAG" \
    $AFF_ARG \
    $EXTRA_ARGS \
    > "$LOG_NAME" 2>&1 &

echo "Launched with log: $LOG_NAME"
