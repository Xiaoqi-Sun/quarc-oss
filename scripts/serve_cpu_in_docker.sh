#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
fi

if [ "$(docker ps -aq -f status=exited -f name=^quarc_service$)" ]; then
  # cleanup if container died;
  # otherwise it would've been handled by make stop already
  docker rm quarc_service
fi

docker run -d \
  --name quarc_service \
  -p 9910:9910 \
  -v "$PWD/configs:/app/quarc/configs" \
  -v "$PWD/checkpoints:/app/quarc/checkpoints" \
  -v "$PWD/data:/app/quarc/data" \
  -t ${ASKCOS_REGISTRY}/quarc:1.0-cpu \
  python quarc_server.py \
  --config-path /app/quarc/configs/hybrid_pipeline_oss.yaml \
  --processed-data-dir /app/quarc/data/processed
