#!/bin/bash

# Set default environment variables if not provided
export RAW_DIR=${RAW_DIR:-"$PWD/data/raw"}
export LOG_DIR=${LOG_DIR:-"$PWD/logs/preprocessing"}
export DUMP_DIR=${DUMP_DIR:-"$PWD/data/interim/dump"}
export GROUPED_DIR=${GROUPED_DIR:-"$PWD/data/interim/grouped"}
export TEMP_DEDUP_DIR=${TEMP_DEDUP_DIR:-"$PWD/data/interim/temp_dedup"}
export SPLIT_DIR=${SPLIT_DIR:-"$PWD/data/interim/split"}
export FINAL_DEDUP_PATH=${FINAL_DEDUP_PATH:-"$PWD/data/interim/final_deduped.pickle"}
export FINAL_DEDUP_FILTERED_PATH=${FINAL_DEDUP_FILTERED_PATH:-"$PWD/data/interim/final_deduped_filtered.pickle"}
export FINAL_DEDUP_FILTERED_WITH_UNCAT_PATH=${FINAL_DEDUP_FILTERED_WITH_UNCAT_PATH:-"$PWD/data/interim/final_deduped_filtered_with_uncat.pickle"}
export AGENT_ENCODER_LIST_PATH=${AGENT_ENCODER_LIST_PATH:-"$PWD/data/processed/agent_encoder/agent_encoder_list.json"}
export AGENT_OTHER_DICT_PATH=${AGENT_OTHER_DICT_PATH:-"$PWD/data/processed/agent_encoder/agent_other_dict.json"}
export CONV_RULES_PATH=${CONV_RULES_PATH:-"$PWD/data/processed/agent_encoder/agent_rules_v1.json"}
export STAGE1_DIR=${STAGE1_DIR:-"$PWD/data/processed/stage1"}
export STAGE2_DIR=${STAGE2_DIR:-"$PWD/data/processed/stage2"}
export STAGE3_DIR=${STAGE3_DIR:-"$PWD/data/processed/stage3"}
export STAGE4_DIR=${STAGE4_DIR:-"$PWD/data/processed/stage4"}

# Set Docker image and container paths
export QUARC_REGISTRY=${QUARC_REGISTRY:-"quarc"}
export CONFIG_PATH=${CONFIG_PATH:-"configs/preprocess_config.yaml"}

# Create necessary directories if they don't exist
mkdir -p $(dirname "$RAW_DIR") $(dirname "$LOG_DIR") $(dirname "$DUMP_DIR") $(dirname "$GROUPED_DIR")
mkdir -p $(dirname "$TEMP_DEDUP_DIR") $(dirname "$SPLIT_DIR") $(dirname "$FINAL_DEDUP_PATH")
mkdir -p $(dirname "$AGENT_ENCODER_LIST_PATH") "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR" "$STAGE4_DIR"

docker run --rm \
    -v "$RAW_DIR":/app/quarc/data/raw \
    -v "$LOG_DIR":/app/quarc/logs \
    -v "$DUMP_DIR":/app/quarc/data/interim/dump \
    -v "$GROUPED_DIR":/app/quarc/data/interim/grouped \
    -v "$TEMP_DEDUP_DIR":/app/quarc/data/interim/temp_dedup \
    -v "$SPLIT_DIR":/app/quarc/data/interim/split \
    -v "$(dirname "$FINAL_DEDUP_PATH")":/app/quarc/data/interim \
    -v "$(dirname "$AGENT_ENCODER_LIST_PATH")":/app/quarc/data/processed/agent_encoder \
    -v "$STAGE1_DIR":/app/quarc/data/processed/stage1 \
    -v "$STAGE2_DIR":/app/quarc/data/processed/stage2 \
    -v "$STAGE3_DIR":/app/quarc/data/processed/stage3 \
    -v "$STAGE4_DIR":/app/quarc/data/processed/stage4 \
    -t "${QUARC_REGISTRY}":latest \
    python scripts/preprocess.py \
    --config=configs/preprocess_config.yaml \
    --all
