#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train a simple vanilla autoencoder without RTM
# export WANDB_MODE=disabled
# python3 -m pdb train.py --config configs/vanilla_AE_scaled.json

# Test the trained model
# python3 -m pdb test.py --config configs/vanilla_AE_scaled.json \
#         --resume saved/models/VanillaAE_scaled/0510_211403/model_best.pth

# Generate synthetic data from RTM
python3 datasets/sampling/sampling.py
