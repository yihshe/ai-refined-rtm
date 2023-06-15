# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
# Assuming df is your DataFrame and 'ground_truth' and 'prediction' are the columns
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM/0612_175828_/'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/VanillaAE_scaled/0612_220221_'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM_syn/0614_112532'
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/NNRegressor/0612_181507'
CSV_PATH = os.path.join(BASE_PATH, 'model_best_testset_analyzer_syn.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'linescatter')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'sd', 'h', 'cd']

df = pd.read_csv(CSV_PATH)

# %%
# plot the linescatter for the ourput and target of each band
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
for i, band in enumerate(S2_BANDS):
    ax = axs[i//4, i % 4]
    # adjust the point size and alpha and color
    sns.scatterplot(x='target_'+band, y='output_'+band,
                    data=df, ax=ax, s=8, alpha=0.5)
    fontsize = 16
    ax.set_title(band, fontsize=fontsize)
    ax.set_xlabel('target', fontsize=fontsize)
    ax.set_ylabel('output', fontsize=fontsize)

    # plot the diagonal line
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_bands.png'))
plt.show()

# %%
# plot the linescatter for the ourput and target of each attribute
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
for i, attr in enumerate(ATTRS):
    ax = axs[i//3, i % 3]
    # adjust the point size and alpha and color
    sns.scatterplot(x='target_'+attr, y='output_'+attr,
                    data=df, ax=ax, s=8, alpha=0.5)
    fontsize = 16
    ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('target', fontsize=fontsize)
    ax.set_ylabel('output', fontsize=fontsize)

    # plot the diagonal line
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_vars.png'))
plt.show()

# %%
