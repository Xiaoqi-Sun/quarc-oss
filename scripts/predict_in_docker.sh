#!/bin/bash

docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_predictor.py \
    --config-path=${PIPELINE_CONFIG_PATH} \
    --input=/app/quarc/data/processed/overlap/overlap_test.pickle \
    --output=/app/quarc/data/processed/overlap/overlap_test_predictions.json \
    --top-k 10