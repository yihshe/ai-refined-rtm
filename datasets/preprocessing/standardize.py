import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import json

# Split the training data further into train and validation sets
# Standardize all datasets according to the training set
CSV_PATH = "/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped_train_valid.csv"
CSV_PATH2 = "/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped_test.csv"
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
SAVE_PATH = '/maps/ys611/ai-refined-rtm/data/real/'
ATTRS = ['class', 'sample_id', 'date']

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# # suffix = "all_CA_range_norm"
# # BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20230715/'
# # SAVE_DIR = os.path.join(BASE_DIR, suffix)
# BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20230816/'
# SAVE_DIR = os.path.join(BASE_DIR, 'synthetic_sys_bias')
# CSV_PATH = os.path.join(SAVE_DIR, "synthetic_train_valid.csv")
# CSV_PATH2 = os.path.join(SAVE_DIR, "synthetic_test.csv")
# S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
#             'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
#             'B12_SWI2']

df = pd.read_csv(CSV_PATH)
df2 = pd.read_csv(CSV_PATH2)

ATTRS = [k for k in df.columns if k not in S2_BANDS+['B01', 'B10']]

rtm_paras = json.load(open("/maps/ys611/ai-refined-rtm/configs/rtm_paras.json"))

n_samples = df.shape[0]
# keep the random seed same as the one used in the training script
idx_full = np.arange(n_samples)
split = 0.2
len_valid = int(n_samples * split)
np.random.seed(0)
np.random.shuffle(idx_full)
valid_idx = idx_full[0:len_valid]
train_idx = np.delete(idx_full, np.arange(0, len_valid))


def standardize(df, df2, columns, data_type='real'):
    # Range of the reflectance should be 0 to 1
    factor = 10000.0 if data_type == 'real' else 1.0
    # TODO: for real data, first scale the data by the factor of 10000
    df_valid = df[columns].iloc[valid_idx]/factor
    df_train = df[columns].iloc[train_idx]/factor
    df_test = df2[columns]/factor

    scaler = preprocessing.StandardScaler().fit(df_train)
    df_train_scaled = scaler.transform(df_train)
    df_valid_scaled = scaler.transform(df_valid)
    df_test_scaled = scaler.transform(df_test)

    return scaler, df_train_scaled, df_valid_scaled, df_test_scaled


def normalize(df, df2, columns):
    """
    No need to normalize if the input attributes are already normalized
    """
    df_train = df[columns].iloc[train_idx]
    df_valid = df[columns].iloc[valid_idx]
    df_test = df2[columns]
    # for col in columns:
    #     min = rtm_paras[col]['min']
    #     max = rtm_paras[col]['max']
    #     df_train[col] = (df_train[col] - min) / (max - min)
    #     df_valid[col] = (df_valid[col] - min) / (max - min)
    #     df_test[col] = (df_test[col] - min) / (max - min)
    return df_train, df_valid, df_test


# standardize spectrums for both real and synthetic datasets
scaler, df_train_scaled, df_valid_scaled, df_test_scaled = standardize(
    df, df2, S2_BANDS, data_type='real')
# normalize rtm parameters for only synthetic datasets
df_train_scaled2, df_valid_scaled2, df_test_scaled2 = normalize(
    df, df2, ATTRS)
# scaler2, df_train_scaled2, df_valid_scaled2, df_test_scaled2 = standardize(
#     df, df2, ATTRS)

df_train_scaled = pd.DataFrame(
    np.hstack((df_train_scaled, df_train_scaled2)), columns=S2_BANDS+ATTRS)
df_valid_scaled = pd.DataFrame(
    np.hstack((df_valid_scaled, df_valid_scaled2)), columns=S2_BANDS+ATTRS)
df_test_scaled = pd.DataFrame(
    np.hstack((df_test_scaled, df_test_scaled2)), columns=S2_BANDS+ATTRS)

np.save(os.path.join(SAVE_PATH, 'train_x_mean.npy'), scaler.mean_)
np.save(os.path.join(SAVE_PATH, 'train_x_scale.npy'), scaler.scale_)
# np.save(os.path.join(BASE_DIR, 'train_y_mean.npy'), scaler2.mean_)
# np.save(os.path.join(BASE_DIR, 'train_y_scale.npy'), scaler2.scale_)

# df_valid = df[S2_BANDS].iloc[valid_idx]
# df_train = df[S2_BANDS].iloc[train_idx]
# df_test = df2[S2_BANDS]

# scaler = preprocessing.StandardScaler().fit(df_train)
# df_train_scaled = scaler.transform(df_train)
# df_valid_scaled = scaler.transform(df_valid)
# df_test_scaled = scaler.transform(df_test)

# df_train_scaled = pd.DataFrame(df_train_scaled, columns=S2_BANDS)
# df_valid_scaled = pd.DataFrame(df_valid_scaled, columns=S2_BANDS)
# df_test_scaled = pd.DataFrame(df_test_scaled, columns=S2_BANDS)

# for attr in ATTRS:
#     df_train_scaled[attr] = df[attr].iloc[train_idx].values
#     df_valid_scaled[attr] = df[attr].iloc[valid_idx].values
#     df_test_scaled[attr] = df2[attr].values

# save the mean and scale of the training set
# np.save('/maps/ys611/ai-refined-rtm/data/train_mean.npy', scaler.mean_)
# np.save('/maps/ys611/ai-refined-rtm/data/train_scale.npy', scaler.scale_)
# np.save(os.path.join(BASE_DIR, 'train_mean.npy'), scaler.mean_)
# np.save(os.path.join(BASE_DIR, 'train_scale.npy'), scaler.scale_)

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
# df_train_scaled.to_csv(
#     os.path.join(SAVE_DIR, 'synthetic_train_scaled.csv'),
#     index=False)
# df_valid_scaled.to_csv(
#     os.path.join(SAVE_DIR, 'synthetic_valid_scaled.csv'),
#     index=False)
# df_test_scaled.to_csv(
#     os.path.join(SAVE_DIR, 'synthetic_test_scaled.csv'),
#     index=False)
print('done')
