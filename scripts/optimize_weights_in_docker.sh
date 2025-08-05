#!/bin/bash

PIPELINE_PATH="configs/test_pipeline"

# Optimize weights using top 5 accuracy
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python pipeline_weight_optimizer.py \
    --pipeline $PIPELINE_PATH \
    --processed-data-dir /app/quarc/data/processed \
    --data-path /app/quarc/data/processed/overlap/overlap_val.pickle \
    --output-dir /app/quarc/data/precomputed/ \
    --n-trials 50 \
    --sample-size 10000 \
    --use-geometric \
    --use-topk 5

# Optimize weights using top 10 accuracy (skipped precomputation)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python pipeline_weight_optimizer.py \
    --pipeline $PIPELINE_PATH \
    --processed-data-dir /app/quarc/data/processed \
    --data-path /app/quarc/data/processed/overlap/overlap_val.pickle \
    --output-dir /app/quarc/data/precomputed/ \
    --skip-precompute \
    --n-trials 50 \
    --sample-size 10000 \
    --use-geometric \
    --use-topk 10

echo "Review the logs and JSON files to see optimal weights."
echo "Update your *_model_config.yaml accordingly for final deployment."