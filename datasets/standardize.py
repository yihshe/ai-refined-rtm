import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import pdb
pdb.set_trace()

CSV_PATH = "/maps/ys611/ai-refined-rtm/data/BPWW_extract_2018_reshaped_train.csv"
CSV_PATH2 = "/maps/ys611/ai-refined-rtm/data/BPWW_extract_2018_reshaped_test.csv"
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
SAVE_PATH = '/maps/ys611/ai-refined-rtm/data/'

ATTRS = ['class', 'sample_id', 'date']
df = pd.read_csv(CSV_PATH)
df2 = pd.read_csv(CSV_PATH2)

n_samples = df.shape[0]
# keep the random seed same as the one used in the training script
idx_full = np.arange(n_samples)
split = 0.2
len_valid = int(n_samples * split)
np.random.seed(0)
np.random.shuffle(idx_full)
valid_idx = idx_full[0:len_valid]
train_idx = np.delete(idx_full, np.arange(0, len_valid))

df_valid = df[S2_BANDS].iloc[valid_idx]
df_train = df[S2_BANDS].iloc[train_idx]
df_test = df2[S2_BANDS]

scaler = preprocessing.StandardScaler().fit(df_train)
df_train_scaled = scaler.transform(df_train)
df_valid_scaled = scaler.transform(df_valid)
df_test_scaled = scaler.transform(df_test)


df_train_scaled = pd.DataFrame(df_train_scaled, columns=S2_BANDS)
df_valid_scaled = pd.DataFrame(df_valid_scaled, columns=S2_BANDS)
df_test_scaled = pd.DataFrame(df_test_scaled, columns=S2_BANDS)
for attr in ATTRS:
    df_train_scaled[attr] = df[attr].iloc[train_idx].values
    df_valid_scaled[attr] = df[attr].iloc[valid_idx].values
    df_test_scaled[attr] = df2[attr].values

# save the mean and scale of the training set
# np.save('/maps/ys611/ai-refined-rtm/data/train_mean.npy', scaler.mean_)
# np.save('/maps/ys611/ai-refined-rtm/data/train_scale.npy', scaler.scale_)

# save the scaled data
df_train_scaled.to_csv(
    os.path.join(SAVE_PATH, 'BPWW_extract_2018_reshaped_train_scaled.csv'),
    index=False)
df_valid_scaled.to_csv(
    os.path.join(SAVE_PATH, 'BPWW_extract_2018_reshaped_valid_scaled.csv'),
    index=False)
df_test_scaled.to_csv(
    os.path.join(SAVE_PATH, 'BPWW_extract_2018_reshaped_test_scaled.csv'),
    index=False)
print('done')
