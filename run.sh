# Train a simple vanilla autoencoder without RTM
# export WANDB_MODE=disabled
python3 -m pdb train.py --config configs/vanilla_AE.json

# Test the trained model
# python3 -m pdb test.py --config configs/vanilla_AE.json \
#         --resume saved/models/VanillaAE/0503_053838/model_best.pth
