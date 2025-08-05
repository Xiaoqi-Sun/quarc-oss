#!/bin/bash

# Put the cleaned ReactionDatum into data/interim and update the config file accordingly

docker run --rm --shm-size=10gb \
    -v "$PWD/data":/app/quarc/data \
    -v "$PWD/logs":/app/quarc/logs \
    -v "$PWD/configs":/app/quarc/configs \
    -v "$FINAL_DEDUP_PATH":/app/quarc/data/interim/final_deduped.pickle \
    -t "${ASKCOS_REGISTRY}/quarc:1.0-gpu" \
    python quarc_processor.py \
    --config=configs/preprocess_config.yaml \
    --generate_vocab \
    --init_filter \
    --split \
    --stage1_filter \
    --stage2_filter \
    --stage3_filter \
    --stage4_filter \