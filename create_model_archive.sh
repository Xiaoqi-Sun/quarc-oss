#!/bin/bash

# QUARC Model Archive Creation Script
# This script packages QUARC models and dependencies into a .mar file for TorchServe

set -e

# Configuration
MODEL_NAME="quarc"
MODEL_VERSION="1.0"
MODEL_TYPE=${1:-"ffn"}  # ffn or gnn
ARCHIVE_DIR="./model_archives"
TEMP_DIR="./temp_archive"

echo "Creating QUARC ${MODEL_TYPE} model archive..."

# Clean up previous runs
rm -rf $TEMP_DIR
rm -rf $ARCHIVE_DIR
mkdir -p $ARCHIVE_DIR
mkdir -p $TEMP_DIR

# Copy handler
cp quarc_handler.py $TEMP_DIR/

# Copy model configuration
if [ "$MODEL_TYPE" = "ffn" ]; then
    cp checkpoints/ffn_pipeline.yaml $TEMP_DIR/
    CONFIG_FILE="ffn_pipeline.yaml"
else
    cp checkpoints/gnn_pipeline.yaml $TEMP_DIR/
    CONFIG_FILE="gnn_pipeline.yaml"
fi

# Copy model checkpoints
echo "Copying model checkpoints..."
cp -r checkpoints/ $TEMP_DIR/

# Copy supporting data files
echo "Copying supporting data files..."
if [ -d "data/processed/agent_encoder" ]; then
    cp data/processed/agent_encoder/agent_encoder_list.json $TEMP_DIR/ 2>/dev/null || echo "Warning: agent_encoder_list.json not found"
    cp data/processed/agent_encoder/agent_rules_v1.json $TEMP_DIR/ 2>/dev/null || echo "Warning: agent_rules_v1.json not found"
    cp data/processed/agent_encoder/agent_other_dict.json $TEMP_DIR/ 2>/dev/null || echo "Warning: agent_other_dict.json not found"
fi

# Create requirements file for the archive
cat > $TEMP_DIR/requirements.txt << EOF
torch>=2.1.0
rdkit
pandas
numpy
scikit-learn
chemprop
EOF

# Create model archive
echo "Creating model archive..."

# Build extra files list
EXTRA_FILES="$TEMP_DIR/$CONFIG_FILE,$TEMP_DIR/checkpoints"
if [ -f "$TEMP_DIR/agent_encoder_list.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,$TEMP_DIR/agent_encoder_list.json"
fi
if [ -f "$TEMP_DIR/agent_rules_v1.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,$TEMP_DIR/agent_rules_v1.json"
fi
if [ -f "$TEMP_DIR/agent_other_dict.json" ]; then
    EXTRA_FILES="$EXTRA_FILES,$TEMP_DIR/agent_other_dict.json"
fi

torch-model-archiver \
    --model-name ${MODEL_NAME}_${MODEL_TYPE} \
    --version $MODEL_VERSION \
    --handler $TEMP_DIR/quarc_handler.py \
    --runtime python \
    --export-path $ARCHIVE_DIR \
    --extra-files "$EXTRA_FILES" \
    --requirements-file $TEMP_DIR/requirements.txt \
    --force

echo "Model archive created: $ARCHIVE_DIR/${MODEL_NAME}_${MODEL_TYPE}.mar"

# Clean up temp directory
rm -rf $TEMP_DIR

echo "Archive creation complete!"
echo ""
echo "To serve the model, run:"
echo "torchserve --start --foreground --ncs --model-store=$ARCHIVE_DIR --models ${MODEL_NAME}_${MODEL_TYPE}=${MODEL_NAME}_${MODEL_TYPE}.mar"
echo ""
echo "To test the service:"
echo "curl -X POST http://localhost:8080/predictions/${MODEL_NAME}_${MODEL_TYPE} \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"reactions\": [{\"rxn_smiles\": \"CC(=O)O.CCO>>CC(=O)OCC.O\", \"rxn_class\": \"1.2.3\"}]}'"