# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# %%
# Original synthetic dataset
# BASE_PATH = '/maps/ys611/ai-refined-rtm/data/synthetic/20230611/'
# CSV_PATH = os.path.join(BASE_PATH, 'synthetic.csv')
# SAVE_PATH = os.path.join(BASE_PATH, 'correlation')
BASE_PATH = '/maps/ys611/ai-refined-rtm/data/real'
CSV_PATH = os.path.join(BASE_PATH, 'BPWW_extract_2018_reshaped.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'correlation')
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/NNRegressor/0612_181507/'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM_syn/0614_112532/'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM/0612_175828_'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM_corr/0615_171950'
# CSV_PATH = os.path.join(BASE_PATH, 'model_best_testset_analyzer.csv')
# SAVE_PATH = os.path.join(BASE_PATH, 'correlation')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'sd', 'h', 'cd']

# df_testset = pd.read_csv(
#     '/maps/ys611/ai-refined-rtm/data/synthetic/20230611/synthetic_test_scaled.csv')
# df_testset = pd.read_csv(
#     '/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped_test_scaled.csv')
df = pd.read_csv(CSV_PATH)
# df = pd.concat([df_testset, df], axis=1)
correlation_matrix = df.corr()
# correlation_matrix = correlation_matrix.loc['latent_N':'latent_cd',
#                                             'B02_BLUE':'B12_SWI2']
# correlation_matrix = correlation_matrix.loc['latent_N':'latent_cd',
#                                             'latent_N':'latent_cd']
correlation_matrix = correlation_matrix.loc[S2_BANDS,
                                            S2_BANDS]


# %%
# Assuming correlation_matrix is your correlation DataFrame
sns.set(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# f, ax = plt.subplots(figsize=(9, 9))

# Generate a custom diverging colormap
cmap = 'coolwarm'

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
            fmt='.2f', annot_kws={'size': 12})

# Set rotation of x and y tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   horizontalalignment='right')
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, 'correlation_matrix_latentvar_v_inputband_real.png'), dpi=300)
# plt.savefig(os.path.join(
#     SAVE_PATH, 'correlation_matrix_latentvar_v_latentvar_real.png'), dpi=300)
plt.savefig(os.path.join(
    SAVE_PATH, 'correlation_matrix_band_v_band.png'), dpi=300)

plt.show()


# %%
