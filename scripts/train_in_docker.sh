#!/bin/bash

export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core

# hparams
AGENT_VOCAB_SIZE=1376
BATCH_SIZE=512
NUM_GPUS=1

# paths
STAGE1_TRAIN_PATH="/app/quarc/data/processed/stage1/stage1_train.pickle"
STAGE1_VAL_PATH="/app/quarc/data/processed/stage1/stage1_val.pickle"
STAGE2_TRAIN_PATH="/app/quarc/data/processed/stage2/stage2_train.pickle"
STAGE2_VAL_PATH="/app/quarc/data/processed/stage2/stage2_val.pickle"
STAGE3_TRAIN_PATH="/app/quarc/data/processed/stage3/stage3_train.pickle"
STAGE3_VAL_PATH="/app/quarc/data/processed/stage3/stage3_val.pickle"
STAGE4_TRAIN_PATH="/app/quarc/data/processed/stage4/stage4_train.pickle"
STAGE4_VAL_PATH="/app/quarc/data/processed/stage4/stage4_val.pickle"

stage 1 (GNN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type gnn \
    --stage 1 \
    --logger-name stage1_agent_gnn \
    --max-epochs 20 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --graph-hidden-size 1024 \
    --depth 2 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --output-size $AGENT_VOCAB_SIZE \
    --num-classes $AGENT_VOCAB_SIZE \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path $STAGE1_TRAIN_PATH \
    --val-data-path $STAGE1_VAL_PATH

# stage 2 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 2 \
    --logger-name stage2_temperature_ffn \
    --max-epochs 5 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 6 \
    --output-size 32 \
    --num-classes $AGENT_VOCAB_SIZE \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path $STAGE2_TRAIN_PATH \
    --val-data-path $STAGE2_VAL_PATH

# stage 3 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 3 \
    --logger-name stage3_reactant_amount_ffn \
    --max-epochs 5 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 2 \
    --output-size 15 \
    --num-classes $AGENT_VOCAB_SIZE \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path $STAGE3_TRAIN_PATH \
    --val-data-path $STAGE3_VAL_PATH

# stage 4 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 4 \
    --logger-name stage4_agent_amount_ffn \
    --max-epochs 5 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --output-size 27 \
    --num-classes $AGENT_VOCAB_SIZE \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path $STAGE4_TRAIN_PATH \
    --val-data-path $STAGE4_VAL_PATH