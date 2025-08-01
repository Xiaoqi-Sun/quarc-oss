#!/bin/bash

# hparams
export BATCH_SIZE=512
export NUM_GPUS=1

# stage 1 (GNN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
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
    --graph-input-dim 1024 \
    --output-size 1376 \
    --num-classes 1376 \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path /app/quarc/data/processed/stage1/stage1_train.pickle \
    --val-data-path /app/quarc/data/processed/stage1/stage1_val.pickle \

# stage 2 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 2 \
    --logger-name stage2_temperature_ffn \
    --max-epochs 30 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 6 \
    --output-size 32 \
    --num-classes 1376 \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path /app/quarc/data/processed/stage2/stage2_train.pickle \
    --val-data-path /app/quarc/data/processed/stage2/stage2_val.pickle \

# stage 3 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 3 \
    --logger-name stage3_reactant_amount_ffn \
    --max-epochs 30` \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 2 \
    --output-size 15 \
    --num-classes 1376 \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path /app/quarc/data/processed/stage3/stage3_train.pickle \
    --val-data-path /app/quarc/data/processed/stage3/stage3_val.pickle \

# stage 4 (FFN)
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python quarc_trainer.py \
    --model-type ffn \
    --stage 4 \
    --logger-name stage4_agent_amount_ffn \
    --max-epochs 30 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr 1e-3 \
    --hidden-size 2048 \
    --n-blocks 3 \
    --output-size 27 \
    --num-classes 1376 \
    --processed-data-dir /app/quarc/data/processed \
    --num-workers 8 \
    --train-data-path /app/quarc/data/processed/stage4/stage4_train.pickle \
    --val-data-path /app/quarc/data/processed/stage4/stage4_val.pickle \

