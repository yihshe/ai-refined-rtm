"""
Run this script to reshape the BPWW_extract_2018.csv file 
The reshaped file will be used for training the AutoEncoder
"""
import os
import numpy as np
import pandas as pd
from datasets.spectrumS2 import SpectrumS2
import data_loader.data_loaders as module_data
import pdb
pdb.set_trace()

TRAIN_DATA_DIR = '/maps/ys611/ai-refined-rtm/data/BPWW_extract_2018_reshaped_train.csv'
TEST_DATA_DIR = '/maps/ys611/ai-refined-rtm/data/BPWW_extract_2018_reshaped_test.csv'
dataset = SpectrumS2(TRAIN_DATA_DIR)
# dataset[0]
train_loader = module_data.SpectrumS2DataLoader(TRAIN_DATA_DIR,
                                                batch_size=1,
                                                shuffle=False,
                                                validation_split=0.0,
                                                num_workers=1)
print(len(train_loader))
