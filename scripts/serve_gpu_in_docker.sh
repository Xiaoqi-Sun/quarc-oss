#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
fi

  # cleanup if container died;
if [ "$(docker ps -aq -f status=exited -f name=^quarc_service$)" ]; then
  docker rm quarc_service
fi

# docker run -d --gpus '"device=0"' \
#   --name quarc_gnn_service \
#   -p 9910:9910 \
#   -v "$PWD/configs:/app/quarc/configs" \
#   -v "$PWD/checkpoints:/app/quarc/checkpoints" \
#   -v "$PWD/data:/app/quarc/data" \
#   -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
#   python quarc_server.py \
#   --config-path /app/quarc/configs/gnn_pipeline.yaml \
#   --processed-data-dir /app/quarc/data/processed

docker run -d --gpus '"device=0"' \
  --name quarc_service \
  -p 9911:9911 \
  -v "$PWD/configs:/app/quarc/configs" \
  -v "$PWD/checkpoints:/app/quarc/checkpoints" \
  -v "$PWD/data:/app/quarc/data" \
  -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
  python quarc_server.py \
  --config-path /app/quarc/configs/ffn_pipeline.yaml \
  --processed-data-dir /app/quarc/data/processed
