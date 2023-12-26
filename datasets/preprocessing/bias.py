"""
This file is used to add random noise to the spectral bands
and also add systematic bias to the spectral bands
"""
# %%
import pandas as pd
import numpy as np
import os
# fix the random seed
np.random.seed(0)

# %%
# load the dataset
BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20230816/'
df = pd.read_csv(os.path.join(BASE_DIR, 'synthetic.csv'))
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
# read the bands from the data and then add random noise sampled from a normal distribution
# with mean 0 and std 0.005, the noise is an array with the same shape as the bands
# then save the bands with noise to a new csv file
df_rand_noise = df.copy()
for band in S2_BANDS:
    df_rand_noise[band] = df_rand_noise[band] + \
        np.random.normal(0, 0.005, len(df_rand_noise))
# fill the negative values with original values
df_rand_noise[df_rand_noise < 0] = df.copy()[df_rand_noise < 0]
# save the results
# df_rand_noise.to_csv(os.path.join(
#     BASE_DIR, 'synthetic_rand_noise.csv'), index=False)

# %%
# add systematic linear bias to the bands
# get the magnitude of each band
df_sys_bias = df_rand_noise.copy()
for band in S2_BANDS:
    # calculate the magnitude of each band
    val = df_rand_noise[band].values
    mag = (sum(val**2)/len(val))**0.5
    # apply the bias to the bands by changing both the slope and the intercept
    # the slope is sampled from a uniform distribution between 0.8 and 1.2
    # the intercept is the scaled magnitude of the band
    slope = np.random.uniform(0.8, 1.2)
    intercept = np.random.uniform(-0.2, 0.2) * mag
    df_sys_bias[band] = df_sys_bias[band] * slope + intercept
    print(band, slope, intercept)
    # fill the negative values with original values
    df_sys_bias[df_sys_bias < 0] = df_rand_noise.copy()[df_sys_bias < 0]
    # save the results
    df_sys_bias.to_csv(os.path.join(
        BASE_DIR, 'synthetic_sys_bias.csv'), index=False)

# %%
