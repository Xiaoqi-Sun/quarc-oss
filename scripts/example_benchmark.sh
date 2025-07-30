#!/bin/bash
export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
export DATA_NAME="example_dataset"

export RAW_DIR="$PWD/data/$DATA_NAME/raw_json"
export LOG_DIR="$PWD/logs/$DATA_NAME"

# Set intermediate processing paths
export DUMP_DIR="$PWD/data/$DATA_NAME/interim/dump"
export GROUPED_DIR="$PWD/data/$DATA_NAME/interim/grouped"
export TEMP_DEDUP_DIR="$PWD/data/$DATA_NAME/interim/temp_dedup"
export SPLIT_DIR="$PWD/data/$DATA_NAME/interim/split"

# Set key intermediate files
export FINAL_DEDUP_PATH="$PWD/data/$DATA_NAME/interim/final_deduped.pickle"
export FINAL_DEDUP_FILTERED_PATH="$PWD/data/$DATA_NAME/interim/final_deduped_filtered.pickle"
export FINAL_DEDUP_FILTERED_WITH_UNCAT_PATH="$PWD/data/$DATA_NAME/interim/final_deduped_filtered_with_uncat.pickle"

# Set agent vocabulary paths
export AGENT_ENCODER_LIST_PATH="$PWD/data/$DATA_NAME/processed/agent_encoder/agent_encoder_list.json"
export AGENT_OTHER_DICT_PATH="$PWD/data/$DATA_NAME/processed/agent_encoder/agent_other_dict.json"
export CONV_RULES_PATH="$PWD/data/$DATA_NAME/processed/agent_encoder/agent_rules_v1.json"

# Set stage-specific output directories
export STAGE1_DIR="$PWD/data/$DATA_NAME/processed/stage1"
export STAGE2_DIR="$PWD/data/$DATA_NAME/processed/stage2"
export STAGE3_DIR="$PWD/data/$DATA_NAME/processed/stage3"
export STAGE4_DIR="$PWD/data/$DATA_NAME/processed/stage4"

# Check that raw data directory exists
[ -d "$RAW_DIR" ] || { echo "Raw data directory $RAW_DIR does not exist"; exit 1; }
mkdir -p $(dirname "$LOG_DIR") $(dirname "$DUMP_DIR") $(dirname "$GROUPED_DIR")
mkdir -p $(dirname "$TEMP_DEDUP_DIR") $(dirname "$SPLIT_DIR") $(dirname "$FINAL_DEDUP_PATH")
mkdir -p $(dirname "$AGENT_ENCODER_LIST_PATH") "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR" "$STAGE4_DIR"


echo "Starting QUARC preprocessing benchmark for dataset: $DATA_NAME"
echo "Raw data directory: $RAW_DIR"
echo "Output will be saved to: data/$DATA_NAME/"

bash scripts/preprocess_in_docker.sh
