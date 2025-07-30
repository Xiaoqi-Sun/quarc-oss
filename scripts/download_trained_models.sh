#!/bin/bash

mkdir -p checkpoints

# Hybrid pipeline models (GNN for agent, FFN for temperature, reactant amount, and agent amount)
GNN_AGENT_MODEL_URL="https://www.dropbox.com/scl/fi/5hxd7ir4wj0p5nuflxpf3/gnn_agent_model_oss.ckpt?rlkey=m9mfxssehx9xe324fhszzpy4y&st=05pescoj&dl=1"
FFN_TEMPERATURE_MODEL_URL="https://www.dropbox.com/scl/fi/v2vswiy4l1hte38e81sij/ffn_temperature_model.ckpt?rlkey=9zajegwnlar3rfokejkizwz6j&st=wx1w4dyd&dl=1"
FFN_REACTANT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/3f1lm61uurdpk30cvrbqo/ffn_reactant_amount_model.ckpt?rlkey=ix869p7i98bg1j5bp1dvt5vqe&st=w1q3rep4&dl=1"
FFN_AGENT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/mt9vsiql4e4p5df7y9adh/ffn_agent_amount_model.ckpt?rlkey=232i309za50w8epgyk29kic3w&st=bib02ht9&dl=1"

# Download GNN agent model
if [ ! -f checkpoints/gnn_agent_model.ckpt ]; then
    echo "checkpoints/gnn_agent_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/gnn_agent_model.ckpt \
      "${GNN_AGENT_MODEL_URL}"
    echo "checkpoints/gnn_agent_model.ckpt Downloaded."
fi

# Download FFN temperature model
if [ ! -f checkpoints/ffn_temperature_model.ckpt ]; then
    echo "checkpoints/ffn_temperature_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/ffn_temperature_model.ckpt \
      "${FFN_TEMPERATURE_MODEL_URL}"
    echo "checkpoints/ffn_temperature_model.ckpt Downloaded."
fi

# Download FFN reactant amount model
if [ ! -f checkpoints/ffn_reactant_amount_model.ckpt ]; then
    echo "checkpoints/ffn_reactant_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/ffn_reactant_amount_model.ckpt \
      "${FFN_REACTANT_AMOUNT_MODEL_URL}"
    echo "checkpoints/ffn_reactant_amount_model.ckpt Downloaded."
fi

# Download FFN agent amount model
if [ ! -f checkpoints/ffn_agent_amount_model.ckpt ]; then
    echo "checkpoints/ffn_agent_amount_model.ckpt not found. Downloading.."
    wget -q --show-progress -O checkpoints/ffn_agent_amount_model.ckpt \
      "${FFN_AGENT_AMOUNT_MODEL_URL}"
    echo "checkpoints/ffn_agent_amount_model.ckpt Downloaded."
fi

echo "All models downloaded to checkpoints/ directory."
