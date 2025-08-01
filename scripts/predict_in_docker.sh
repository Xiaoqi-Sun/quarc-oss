#!/bin/bash

docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \

    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_predictor.py \
    --config=configs/hybrid_pipeline_oss.yaml \
    --input=/app/quarc/data/external/test_reactions.json \
    --output=/app/quarc/data/external/test_reactions_predictions.json