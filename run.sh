#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train a simple vanilla autoencoder without RTM
# export WANDB_MODE=disabled
# python3 -m pdb train.py --config configs/vanilla_AE_scaled.json

# Test the trained model
# python3 -m pdb test.py --config configs/vanilla_AE_scaled.json \
#         --resume saved/models/VanillaAE_scaled/0510_211403/model_best.pth

# python3 -m pdb test.py --config configs/AE_RTM.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM/0607_164242/model_best.pth

# Generate synthetic data from RTM
# python3 datasets/sampling/sampling.py

# Split the data into train and test
# python3 datasets/preprocessing/train_test_split.py

# Standardize the data
# python3 datasets/preprocessing/standardize.py

# Train NN Regressor
# python3 train.py --config configs/NN_regressor.json 

# Train a simple vanilla autoencoder with RTM
# python3 train.py --config configs/AE_RTM.json 

# # Train a simple vanilla autoencoder without RTM
# python3 train.py --config configs/vanilla_AE_scaled.json

# Train an autoencoder with RTM and with bias correction
python3 train.py --config configs/AE_RTM_corr.json 


