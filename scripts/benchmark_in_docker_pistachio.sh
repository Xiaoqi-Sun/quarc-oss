#!/bin/bash

export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2

# Specify this if preprocessing from pistachio
export RAW_DIR="REPLACE WITH PISTACHIO EXTRACTED FOLDER"
export TEST_FILE= ...



[ -f $RAW_DIR ] || { echo $RAW_DIR does not exist; exit; }

# Preprocess, will overwrite the existing agent encoder files
bash scripts/preprocess_in_docker_pistachio.sh 

# Train, use the newly generated agent encoder files by default
bash scripts/train_in_docker.sh

# Predict, use the newly generated agent encoder files by default
bash scripts/predict_in_docker.sh
