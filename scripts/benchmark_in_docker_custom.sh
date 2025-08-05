#!/bin/bash

export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2

# Specify this if preprocessing from pistachio
export FINAL_DEDUP_PATH= /path/to/final_deduped.pickle

# Preprocess, will overwrite the existing agent encoder files
bash scripts/preprocess_in_docker_custom.sh

# Train, use the newly generated agent encoder files by default
bash scripts/train_in_docker.sh

# Predict, use the newly generated agent encoder files by default
bash scripts/predict_in_docker.sh
