#!/bin/bash

mkdir -p checkpoints/agent
mkdir -p checkpoints/temperature
mkdir -p checkpoints/reactant_amount
mkdir -p checkpoints/agent_amount

if [ ! -f checkpoints/agent/ffn_agent_model.ckpt ]; then
    echo "checkpoints/agent/ffn_agent_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/agent/ffn_agent_model.ckpt \
      "${FFN_AGENT_MODEL_URL}"
    echo "checkpoints/agent/ffn_agent_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/agent/gnn_agent_model.ckpt ]; then
    echo "checkpoints/agent/gnn_agent_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/agent/gnn_agent_model.ckpt \
      "${GNN_AGENT_MODEL_URL}"
    echo "checkpoints/agent/gnn_agent_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/temperature/ffn_temperature_model.ckpt ]; then
    echo "checkpoints/temperature/ffn_temperature_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/temperature/ffn_temperature_model.ckpt \
      "${FFN_TEMPERATURE_MODEL_URL}"
    echo "checkpoints/temperature/ffn_temperature_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/temperature/gnn_temperature_model.ckpt ]; then
    echo "checkpoints/temperature/gnn_temperature_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/temperature/gnn_temperature_model.ckpt \
      "${GNN_TEMPERATURE_MODEL_URL}"
    echo "checkpoints/temperature/gnn_temperature_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt ]; then
    echo "checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt \
      "${FFN_REACTANT_AMOUNT_MODEL_URL}"
    echo "checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt ]; then
    echo "checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt \
      "${GNN_REACTANT_AMOUNT_MODEL_URL}"
    echo "checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/agent_amount/ffn_agent_amount_model.ckpt ]; then
    echo "checkpoints/agent_amount/ffn_agent_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/agent_amount/ffn_agent_amount_model.ckpt \
      "${FFN_AGENT_AMOUNT_MODEL_URL}"
    echo "checkpoints/agent_amount/ffn_agent_amount_model.ckpt Downloaded."
fi

if [ ! -f checkpoints/agent_amount/gnn_agent_amount_model.ckpt ]; then
    echo "checkpoints/agent_amount/gnn_agent_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/agent_amount/gnn_agent_amount_model.ckpt \
      "${GNN_AGENT_AMOUNT_MODEL_URL}"
    echo "checkpoints/agent_amount/gnn_agent_amount_model.ckpt Downloaded."
fi

# # Copy pipeline configuration files
# if [ ! -f checkpoints/ffn_pipeline.yaml ]; then
#     echo "Copying ffn_pipeline.yaml..."
#     cp ../quarc/checkpoints/ffn_pipeline.yaml checkpoints/
# fi

# if [ ! -f checkpoints/gnn_pipeline.yaml ]; then
#     echo "Copying gnn_pipeline.yaml..."
#     cp ../quarc/checkpoints/gnn_pipeline.yaml checkpoints/
# fi

echo "All checkpoints downloaded successfully!"
