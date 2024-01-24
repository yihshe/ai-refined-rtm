"""
Create a pipeline for preprocessing the synthetic dataset.
Step 1: split the data into train, validation, and test sets
Step 2: standardize the spectra and normalize the variables
"""
#%%
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%%
# fix random seeds for reproducibility
SEED = 123
np.random.seed(SEED)
BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20240124'
DATA_DIR = os.path.join(BASE_DIR, 'synthetic.csv')
SAVE_DIR = BASE_DIR

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
rtm_paras = json.load(open('/maps/ys611/ai-refined-rtm/configs/rtm_paras.json'))
#%% load and split the synthetic data
df = pd.read_csv(DATA_DIR)
# drop the 'B01' and 'B10' bands
df.drop(columns=['B01', 'B10'], inplace=True)
SPLIT_RATIO = 0.2
train, test = train_test_split(
    df, test_size=SPLIT_RATIO, random_state=42)
train, valid = train_test_split(
    train, test_size=SPLIT_RATIO, random_state=42)
#%% standardize the spectra
scaler = preprocessing.StandardScaler().fit(train[S2_BANDS])
train[S2_BANDS] = scaler.transform(train[S2_BANDS])
valid[S2_BANDS] = scaler.transform(valid[S2_BANDS])
test[S2_BANDS] = scaler.transform(test[S2_BANDS])
#%% normalize the variables
for para_name in rtm_paras.keys():
    min = rtm_paras[para_name]['min']
    max = rtm_paras[para_name]['max']
    train[para_name] = (train[para_name] - min)/(max - min)
    valid[para_name] = (valid[para_name] - min)/(max - min)
    test[para_name] = (test[para_name] - min)/(max - min)
#%% save the preprocessed data
np.save(os.path.join(SAVE_DIR, 'train_x_mean.npy'), scaler.mean_)
np.save(os.path.join(SAVE_DIR, 'train_x_scale.npy'), scaler.scale_)
train.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
valid.to_csv(os.path.join(SAVE_DIR, 'valid.csv'), index=False)
test.to_csv(os.path.join(SAVE_DIR, 'test.csv'), index=False)