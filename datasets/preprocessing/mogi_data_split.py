#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import json

#%%
BASE_DIR = '/maps/ys611/ai-refined-rtm/'
DATA_DIR = os.path.join(BASE_DIR, 'data/mogi/ts_filled_ICA9comp.csv')

SAVE_DIR = os.path.join(BASE_DIR, 'data/mogi/seq')
SAVE_DIR_TRAIN = os.path.join(SAVE_DIR, 'train.csv')
SAVE_DIR_VALID = os.path.join(SAVE_DIR, 'valid.csv')
SAVE_DIR_TEST = os.path.join(SAVE_DIR, 'test.csv')

SPLIT_RATIO = 0.2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

data = pd.read_csv(DATA_DIR)
station_info = json.load(open('/maps/ys611/ai-refined-rtm/configs/mogi/station_info.json'))

#%% split the data into train, valid, and test sets
train_valid, test = train_test_split(
    data, test_size=SPLIT_RATIO, random_state=42, shuffle=False)
# train, valid = train_test_split(
#     train_valid, test_size=SPLIT_RATIO, random_state=42, shuffle=False)
valid, train = train_test_split(
    train_valid, test_size=1-SPLIT_RATIO+0.05, random_state=42, shuffle=False)

#%% standardize the dataset
scaler = preprocessing.StandardScaler().fit(train.iloc[:, :36])
train.iloc[:, :36] = scaler.transform(train.iloc[:, :36])
valid.iloc[:, :36] = scaler.transform(valid.iloc[:, :36])
test.iloc[:, :36] = scaler.transform(test.iloc[:, :36])

#%% save the train, valid, and test sets
train.to_csv(SAVE_DIR_TRAIN, index=False)
valid.to_csv(SAVE_DIR_VALID, index=False)
test.to_csv(SAVE_DIR_TEST, index=False)

#%% save the scaler
np.save(os.path.join(SAVE_DIR, 'train_x_mean.npy'), scaler.mean_)
np.save(os.path.join(SAVE_DIR, 'train_x_scale.npy'), scaler.scale_)


# %%
