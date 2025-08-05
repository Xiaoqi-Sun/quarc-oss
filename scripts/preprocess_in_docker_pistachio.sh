#!/bin/bash

docker run --rm --shm-size=10gb \
    -v "$RAW_DIR":/app/quarc/data/raw \
    -v "$PWD/data":/app/quarc/data \
    -v "$PWD/logs":/app/quarc/logs \
    -v "$PWD/configs":/app/quarc/configs \
    -t "${ASKCOS_REGISTRY}/quarc:1.0-gpu" \
    python quarc_processor.py \
    --config=configs/preprocess_config.yaml \
    --run_all