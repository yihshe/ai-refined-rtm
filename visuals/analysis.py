# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# %%
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/'
CSV_PATH1 = os.path.join(
    BASE_PATH, 'AE_RTM/0115_105753/model_best_testset_analyzer.csv')
CSV_PATH2 = os.path.join(
    BASE_PATH, 'AE_RTM_corr/0115_135353/model_best_testset_analyzer.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr/0115_135353/histograms')
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
rtm_paras = json.load(open('/maps/ys611/ai-refined-rtm/configs/rtm_paras.json'))
ATTRS = list(rtm_paras.keys()) + ['cd', 'h']

NUM_BINS = 100
# mkdir if the save path does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read the csv file
df1 = pd.read_csv(CSV_PATH1)
df2 = pd.read_csv(CSV_PATH2)
# retrieve the target and output bands to original scale
MEAN = np.load('/maps/ys611/ai-refined-rtm/data/real/train_x_mean.npy')
SCALE = np.load('/maps/ys611/ai-refined-rtm/data/real/train_x_scale.npy')
for x in ['target', 'output']:
    df1[[f'{x}_{band}' for band in S2_BANDS]] = df1[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN
    df2[[f'{x}_{band}' for band in S2_BANDS]] = df2[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN

# get the unique tree species
coniferous = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra', 
              'Larix decidua', 'Pinus sylvestris']
deciduous = ['Prunus spp', 'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp', 
             'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']

# %%
"""
Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
"""
# Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
NUM_BINS = 100
# create one figure and plot both variable predictions of different models as a subplot
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
for i, attr in enumerate(ATTRS):
    ax=axs[i//3, i % 3]
    sns.histplot(
        df1[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        label='AE_RTM',
        alpha=0.5,
    )
    sns.histplot(
        df2[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        # label='AE_RTM_syn',
        label='AE_RTM_corr',
        alpha=0.6,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, 'histogram_real_testset_rtm_vars_v_AE_RTM_orig_scale_all.png'), dpi=300)
plt.show()

# %%
"""
Histogram of the latent variables between Coniferous and Deciduous for AE_RTM_corr
"""
dfs = {
    'Coniferous': df2[df2['class'].isin(coniferous)],
    'Deciduous': df2[df2['class'].isin(deciduous)],
}

# TODO finish this code
# Histogram of the latent variables of selected species
NUM_BINS = 50
# create one figure and plot both variable predictions of different models as a subplot
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
for i, attr in enumerate(ATTRS):
    ax=axs[i//3, i % 3]
    sns.histplot(
        df_filtered[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        label='AE_RTM_corr',
        alpha=0.6,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, 'histogram_real_testset_rtm_vars_v_AE_RTM_orig_scale_all.png'), dpi=300)
plt.show()
    
# %%
"""
Histogram of the latent variables per tree species for AE_RTM_corr
"""
# Histogram of the latent variables for selected species 
# Plot histogram of biophysical variables of each single species
for species in coniferous+deciduous:
    df_filtered = df2[df2['class']==species]
    # Histogram of the latent variables of selected species
    NUM_BINS = 50
    # create one figure and plot both variable predictions of different models as a subplot
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    for i, attr in enumerate(ATTRS):
        ax=axs[i//3, i % 3]
        sns.histplot(
            df_filtered[f'latent_{attr}'].values,
            bins=NUM_BINS,
            ax=ax,
            color='blue',
            label='AE_RTM_corr',
            alpha=0.6,
        )
        # change the fontsize of the x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=16)
        fontsize = 18
        ax.set_xlabel(attr, fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
        # ax.legend(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(os.path.join(
    #     SAVE_PATH, 'histogram_real_testset_rtm_vars_v_AE_RTM_orig_scale_all.png'), dpi=300)
    plt.show()
    

# %%
"""
Scatter plot of input and reconstruction bands
"""
# Scatter plot of the input and reconstructed bands
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
for i, band in enumerate(S2_BANDS):
    ax = axs[i//4, i % 4]
    # adjust the point size and alpha and color
    sns.scatterplot(x='target_'+band, y='output_'+band,
                    data=df2, ax=ax, s=8, alpha=0.5)
    fontsize = 18
    # set the distance between the title and the plot
    ax.set_title(band, fontsize=fontsize, pad=10)
    ax.set_xlabel('Input', fontsize=fontsize)
    ax.set_ylabel('Reconstruction', fontsize=fontsize)
    # set the same ticks for both x and y axes
    ax.tick_params(axis='both', which='major', labelsize=16)
    # plot the diagonal line
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
# make the last subplot empty
# axs[2, 3].axis('off')
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, 'linescatter_bands.png'))
plt.show()

