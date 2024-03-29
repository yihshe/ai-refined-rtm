#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train a simple vanilla autoencoder without RTM
# export WANDB_MODE=disabled
# python3 -m pdb train.py --config configs/vanilla_AE_scaled.json

# Test the trained model
# python3 -m pdb test.py --config configs/vanilla_AE.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/VanillaAE/0115_104855/model_best.pth
# python3 test_AE_analyze.py --config configs/vanilla_AE.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/VanillaAE/0125_165304/model_best.pth

# Test the trained model of AE with RTM
# python3 -m pdb test_AE_analyze.py --config configs/AE_RTM.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM/0125_170921/model_best.pth
        
# Test the trained model of vanilla AE
# python3 -m pdb test_AE_analyze.py --config configs/vanilla_AE_scaled.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/VanillaAE_scaled/0612_220221/model_best.pth

# Test the trained model of NN regressor on synthetic data
# python3 -m pdb test_NN_analyze.py --config configs/NN_regressor.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/NNRegressor/0124_160519/model_best.pth

# Test the trained model of NN Regressor on real test data
# python3 -m pdb test_NN_analyze.py --config configs/NN_regressor_infer.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/NNRegressor/0124_160519/model_best.pth

# Test the trained model of AE_RTM_syn on synthetic data
# python3 test_AE_syn_analyze.py --config configs/AE_RTM_syn.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM_syn_noised_data/0913_075425/model_best.pth

# python3 test_AE_syn_analyze.py --config configs/AE_RTM_corr.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM_corr_syn_noised_data/0913_112305/model_best.pth

# Test the trained model of AE_RTM_syn on real test data
# python3 -m pdb test_AE_syn_analyze.py --config configs/AE_RTM_syn_infer.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM_syn/0614_112532/model_best.pth

# Test the trained model of AE_RTM_corr on real test data
python3 -m pdb test_AE_analyze.py --config configs/AE_RTM_corr.json \
        --resume /maps/ys611/ai-refined-rtm/saved/models/AE_RTM_corr/0124_000330/model_best.pth

# Generate synthetic data from RTM
# python3 -m pdb datasets/sampling/sampling.py

# Split the data into train and test
# python3 datasets/preprocessing/train_test_split.py

# Standardize the data
# python3 datasets/preprocessing/standardize.py

# Train NN Regressor
# python3 train.py --config configs/NN_regressor.json 

# python3 -m pdb test_NN_analyze.py --config configs/NN_regressor_infer.json \
#         --resume /maps/ys611/ai-refined-rtm/saved/models/NNRegressor_cd/0622_215152/checkpoint-epoch30.pth

# python3 datasets/preprocessing/reshape.py

# Train a simple vanilla autoencoder without RTM
# python3 train.py --config configs/vanilla_AE.json

# Train a simple vanilla autoencoder with RTM
# python3 train.py --config configs/AE_RTM_con.json 

# Train an autoencoder with RTM and with bias correction
# python3 train.py --config configs/AE_RTM_corr_con.json 

# python3 train.py --config configs/vanilla_AE.json

# Train a simple vanilla autoencoder with RTM on synthetic data
# python3 train.py --config configs/AE_RTM_syn.json

# Run unittest
# python3 -m pdb rtm_unittest.py

# Visual analysis of the stabilizer
# python3 -m pdb visuals/training_logs.py

# python3 -m pdb datasets/preprocessing/pipeline_real.py

