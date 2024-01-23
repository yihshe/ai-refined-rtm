"""
Create a pipeline for preprocessing the data. 
Step 1: split the data into train, validation, and test sets
Step 2: standardize the data
Step 3: reshape the data by breaking the time series into multiple samples
"""
#%%
import os
import pandas as pd
import numpy as np
#%%
BASE_DIR = '/maps/ys611/ai-refined-rtm/data/real/'
DATA_DIR = os.path.join(BASE_DIR, 'BPWW_extract_2018.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'BPWW_extract_2018_split')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
