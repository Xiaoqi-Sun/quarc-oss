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
  -p 9610:9610 \
  -t ${ASKCOS_REGISTRY}/context_recommender/quarc:1.0-cpu