#!/bin/bash

# If model configuration is changed, update accordingly:
CHECKPOINT_DIR="/app/quarc/checkpoints/GNN/stage1/stage1_agent_gnn/version_0"
DEPTH=2
GRAPH_HIDDEN_SIZE=1024
N_BLOCKS=3
HIDDEN_SIZE=2048


docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PWD/logs:/app/quarc/logs" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
    python stage1_model_evaluation.py \
    --checkpoint-dir $CHECKPOINT_DIR \
    --test-data-path /app/quarc/data/processed/stage1/stage1_val.pickle \
    --processed-data-dir /app/quarc/data/processed \
    --chunk-size 500 \
    --n-processes 16 \
    --depth $DEPTH \
    --graph-hidden-size $GRAPH_HIDDEN_SIZE \
    --n-blocks $N_BLOCKS \
    --hidden-size $HIDDEN_SIZE

echo ""
echo "Check logs/GNN_model_selection_no_rxnclass.log for results and select best checkpoint"
echo "Update hybrid_pipeline_oss.yaml accordingly"
echo ""