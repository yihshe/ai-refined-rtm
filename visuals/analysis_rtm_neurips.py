# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import json

# %%
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/'
# TODO add VanillaAE and NNRegressor for analysis
CSV_PATH0 = os.path.join(
    BASE_PATH, 'VanillaAE/0123_173930/model_best_testset_analyzer.csv')
CSV_PATH1 = os.path.join(
    BASE_PATH, 'AE_RTM/0123_175312/model_best_testset_analyzer.csv')
CSV_PATH2 = os.path.join(
    BASE_PATH, 'AE_RTM_corr/0124_000330_/model_best_testset_analyzer.csv')
CSV_PATH3 = os.path.join(
    BASE_PATH, 'NNRegressor/0124_160519/model_best_testset_analyzer_real.csv')

# NOTE models trained with embedded conditions on the same data
# CSV_PATH1 = os.path.join(
#     BASE_PATH, 'AE_RTM_con/0201_140847/model_best_testset_analyzer.csv')
# CSV_PATH2 = os.path.join(
#     BASE_PATH, 'AE_RTM_corr_con/0201_201257/model_best_testset_analyzer.csv')

SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr/0124_000330_/plots/neurips521')
# SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr_con/0201_201257/plots')

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
S2_names = {
    'B02_BLUE': 'B2', 'B03_GREEN': 'B3', 'B04_RED': 'B4', 'B05_RE1': 'B5',
    'B06_RE2': 'B6', 'B07_RE3': 'B7', 'B08_NIR1': 'B8', 'B8A_NIR2': 'B8a',
    'B09_WV': 'B9', 'B11_SWI1': 'B11', 'B12_SWI2': 'B12'
}
rtm_paras = json.load(open('/maps/ys611/ai-refined-rtm/configs/rtm_paras.json'))
ATTRS = list(rtm_paras.keys())
# for each attr in ATTRS, create a LaTex variable name like $Z_{\mathrm{attr}}$
ATTRS_LATEX = {
    'N': '$Z_{\mathrm{N}}$', 'cab': '$Z_{\mathrm{cab}}$', 'cw': '$Z_{\mathrm{cw}}$',
    'cm': '$Z_{\mathrm{cm}}$', 'LAI': '$Z_{\mathrm{LAI}}$', 'LAIu': '$Z_{\mathrm{LAIu}}$',
    'fc': '$Z_{\mathrm{fc}}$'
    }
ATTRS_VANILLA = ['1', '2', '3', '4', '5', '6', '7']

NUM_BINS = 100
# mkdir if the save path does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read the csv file
df0 = pd.read_csv(CSV_PATH0)
df1 = pd.read_csv(CSV_PATH1)
df2 = pd.read_csv(CSV_PATH2)
df3 = pd.read_csv(CSV_PATH3)
# retrieve the target and output bands to original scale
MEAN = np.load('/maps/ys611/ai-refined-rtm/data/real/train_x_mean.npy')
SCALE = np.load('/maps/ys611/ai-refined-rtm/data/real/train_x_scale.npy')
for x in ['target', 'output']:
    df1[[f'{x}_{band}' for band in S2_BANDS]] = df1[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN
    df2[[f'{x}_{band}' for band in S2_BANDS]] = df2[[f'{x}_{band}' for band in S2_BANDS]]*SCALE + MEAN
# df2[[f'bias_{band}' for band in S2_BANDS]] = df2[[f'bias_{band}' for band in S2_BANDS]]*SCALE + MEAN
# df2[[f'init_output_{band}' for band in S2_BANDS]] = df2[[f'init_output_{band}' for band in S2_BANDS]]*SCALE + MEAN
    
# drop the columns in df2 beginning with 'l2' and save the data
# df2 = df2[df2.columns.drop(list(df2.filter(regex='l2')))]
# df2.to_csv(os.path.join(SAVE_PATH, 'model_best_testset_analyzer_cleaned.csv'), index=False)

# map the output varibles of df3 to the original scale
for attr in ATTRS:
    df3[f'latent_{attr}'] = df3[f'latent_{attr}']*(
        rtm_paras[attr]['max'] - rtm_paras[attr]['min']) + rtm_paras[attr]['min']
MEAN_SYN = np.load('/maps/ys611/ai-refined-rtm/data/synthetic/20240124/train_x_mean.npy')
SCALE_SYN = np.load('/maps/ys611/ai-refined-rtm/data/synthetic/20240124/train_x_scale.npy')
for x in ['target', 'output']:
    df3[[f'{x}_{band}' for band in S2_BANDS]] = df3[[f'{x}_{band}' for band in S2_BANDS]]*SCALE_SYN + MEAN_SYN
    
# get the unique tree species
coniferous = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra', 
              'Larix decidua', 'Pinus sylvestris']
deciduous = ['Prunus spp', 'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp', 
             'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']
dates = ['2018.04.08', '2018.04.21', '2018.05.06', '2018.07.02', '2018.08.09', 
         '2018.08.21', '2018.08.29', '2018.09.13', '2018.09.18', '2018.09.28', 
         '2018.09.30', '2018.10.05', '2018.10.10', '2018.10.30']

def r_square(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# %%
"""
Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
N2, NNRegressor and AE_RTM_corr
"""
# Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
NUM_BINS = 100
ATTRS = list(rtm_paras.keys())
# ATTRS = ['cw', 'LAI', 'LAIu', 'fc']
# ATTRS = ['N', 'fc']
# create one figure and plot both variable predictions of different models as a subplot
fig, axs = plt.subplots(2,4, figsize = (26, 10))
# fig, axs = plt.subplots(1,4, figsize = (26, 5))
# fig, axs = plt.subplots(1, 2, figsize = (12.5, 5))
for i, attr in enumerate(ATTRS):
    ax=axs[i//4, i % 4]
    # ax = axs[i]
    # sns.histplot(
    #     df1[f'latent_{attr}'].values,
    #     bins=NUM_BINS,
    #     ax=ax,
    #     color='red',
    #     label='w/o $\mathbf{C}$',
    #     alpha=0.5,
    # )
    sns.histplot(
        df3[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        label='NNRegressor',
        alpha=0.5,
    )
    # sns.histplot(
    #     df2[f'latent_{attr}'].values,
    #     bins=NUM_BINS,
    #     ax=ax,
    #     color='blue',
    #     label='w/ $\mathbf{C}$',
    #     alpha=0.5,
    # )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=25)
    # set the range of x axis as the physical range of the variable
    ax.set_xlim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
    fontsize = 30
    ax.set_xlabel(ATTRS_LATEX[attr], fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # set the distance between the y label and the y axis
    ax.yaxis.labelpad = 10
    # ax.legend(fontsize=fontsize-5)
# remove the last subplot
axs[-1, -1].axis('off')
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, 'histogram_realset_vars_ae_rtm_corr_v_wocorr.png'), dpi=300)
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_NN.png'), dpi=300)

plt.show()

# %%
"""
Histogram of the latent variables between Coniferous and Deciduous 
for AE_RTM_corr
"""
# Histogram of the latent variables of selected species
df_coniferous = df2[df2['class'].isin(coniferous)]
df_deciduous = df2[df2['class'].isin(deciduous)]
NUM_BINS = 100
ATTRS = list(rtm_paras.keys())
# ATTRS = ['N', 'cab', 'LAIu', 'fc']
# ATTRS = ['N', 'fc']
# create one figure and plot both variable predictions of different models as a subplot
fig, axs = plt.subplots(2, 4, figsize=(26, 10))
# fig, axs = plt.subplots(1, 4, figsize=(25, 5))
# fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
for i, attr in enumerate(ATTRS):
    ax=axs[i//4, i % 4]
    # ax = axs[i]
    sns.histplot(
        df_coniferous[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        label='Coniferous',
        alpha=0.5,
    )
    sns.histplot(
        df_deciduous[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        # label='AE_RTM_syn',
        label='Deciduous',
        alpha=0.5,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=25)
    fontsize = 30
    ax.set_xlabel(ATTRS_LATEX[attr], fontsize=fontsize)
    ax.set_xlim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # set the distance between the y label and the y axis
    ax.yaxis.labelpad = 10
    ax.legend(fontsize=fontsize-5)
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_corr_coniferous_v_deciduous.png'), dpi=300)
plt.show()
    
# %%
"""
Histogram of the latent variables per tree species for AE_RTM_corr
"""
# Histogram of the latent variables for selected species 
# Plot histogram of biophysical variables of each single species
# NOTE SKIP plot the spectral signature for each species before and after correction
for species in coniferous+deciduous:
    df_filtered = df2[df2['class']==species]
    # Histogram of the latent variables of selected species
    NUM_BINS = 50
    # create one figure and plot both variable predictions of different models as a subplot
    fig, axs = plt.subplots(2, 4, figsize=(25, 10))
    for i, attr in enumerate(ATTRS):
        ax=axs[i//4, i % 4]
        sns.histplot(
            df_filtered[f'latent_{attr}'].values,
            bins=NUM_BINS,
            ax=ax,
            color='blue',
            label='AE_RTM_corr',
            alpha=0.6,
        )
        # change the fontsize of the x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=25)
        fontsize = 30
        ax.set_xlabel(attr, fontsize=fontsize)
        ax.set_xlim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
        ax.set_ylabel('Frequency', fontsize=fontsize)
        # set the distance between the y label and the y axis
        ax.yaxis.labelpad = 10
        # ax.legend(fontsize=fontsize)
    axs[-1, -1].axis('off')
    # Set the title of the figure
    forest_type = 'Coniferous' if species in coniferous else 'Deciduous'
    plt.suptitle(f"{forest_type}: {species}", fontsize=22)
    # Set the distance between the title and the plot
    plt.subplots_adjust(top=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'histogram_realset_vars_corr_{species}.png'), dpi=300)
    plt.show()
# %% N2 histogram of latent variables learned by VanillaAE
NUM_BINS = 100
fig, axs = plt.subplots(2, 4, figsize=(25, 10))
for i, attr in enumerate(ATTRS_VANILLA):
    ax=axs[i//4, i % 4]
    sns.histplot(
        df0[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        label='VanillaAE',
        alpha=0.5,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_vanilla.png'), dpi=300)
#%% N2 histogram of bias learned by AE_RTM_corr
NUM_BINS = 100
fig, axs = plt.subplots(3, 4, figsize=(25, 15))
for i, band in enumerate(S2_BANDS):
    ax=axs[i//4, i % 4]
    sns.histplot(
        df2[f'bias_{band}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        label='AE_RTM_corr',
        alpha=0.5,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=25)
    fontsize = 30
    ax.set_xlabel('Bias', fontsize=fontsize)
    # set 0 as the middle of the x axis by getting the maximum absolute value of the bias
    ax.set_xlim(-np.max(np.abs(df2[f'bias_{band}'])), np.max(np.abs(df2[f'bias_{band}'])))
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.set_title(S2_names[band], fontsize=fontsize)
    ax.yaxis.labelpad = 10
    # ax.legend(fontsize=fontsize)
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_bias_corr.png'), dpi=300)

# %% histogram of bias clustered by forest type for AE_RTM_corr
NUM_BINS = 100
fig, axs = plt.subplots(3, 4, figsize=(25, 15))
df_coniferous = df2[df2['class'].isin(coniferous)]
df_deciduous = df2[df2['class'].isin(deciduous)]
for i, band in enumerate(S2_BANDS):
    ax=axs[i//4, i % 4]
    sns.histplot(
        df_coniferous[f'bias_{band}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        label='Coniferous',
        alpha=0.6,
    )
    sns.histplot(
        df_deciduous[f'bias_{band}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        label='Deciduous',
        alpha=0.6,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=25)
    fontsize = 30
    ax.set_xlabel('Corrected Bias', fontsize=fontsize)
    ax.set_xlim(-np.max(np.abs(df2[f'bias_{band}'])), np.max(np.abs(df2[f'bias_{band}'])))
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.yaxis.labelpad = 10
    ax.set_title(S2_names[band], fontsize=fontsize)
    ax.legend(fontsize=22)
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_bias_corr_coniferous_v_deciduous.png'), dpi=300)
# %%
"""
Scatter plot for selected variables for AE_RTM_corr
"""    
# subplots for fc v. cd, fc v. h
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
var_pairs = [['fc', 'cd'], ['fc', 'h']]
for i, var_pair in enumerate(var_pairs):
    ax = axs[i]
    sns.scatterplot(x=f'latent_{var_pair[0]}', y=f'latent_{var_pair[1]}',
                    data=df2, ax=ax, s=8, alpha=0.5)
    fontsize = 25
    # set the distance between the title and the plot
    # ax.set_title(f'{var_pair[0]} v. {var_pair[1]}', fontsize=fontsize, pad=10)
    ax.set_xlabel(var_pair[0], fontsize=fontsize)
    ax.set_ylabel(var_pair[1], fontsize=fontsize)
    # set the same ticks for both x and y axes
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'scatter_realset_vars_corr_fc_v_cd_h.png'))

# %%
"""
Scatter plot for variable pairs for VanillaAE, AE_RTM, AE_RTM_corr, and NNRegressor
"""
fig, axs = plt.subplots(7, 7, figsize=(25, 25))
# drop the last two elements in ATTRS (cd and h)
# ATTRS2 = ATTRS_VANILLA
ATTRS2 = rtm_paras.keys()
for i, attr1 in enumerate(ATTRS2):
    for j, attr2 in enumerate(ATTRS2):
        ax = axs[i, j]
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df2, ax=ax, s=8, alpha=0.5)
        fontsize = 22
        # set the distance between the title and the plot
        # ax.set_title(f'{attr1} v. {attr2}', fontsize=fontsize, pad=10)
        ax.set_xlabel(ATTRS_LATEX[attr1], fontsize=fontsize)
        ax.set_ylabel(ATTRS_LATEX[attr2], fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'scatter_realset_vars_pairs_corr.png'))
# %%
"""
Scatter plot for variable pairs distinguishing Coniferous and Diceduous 
for VanillaAE, and AE_RTM_corr
"""
fig, axs = plt.subplots(7, 6, figsize=(25, 25))
# drop the last two elements in ATTRS (cd and h)
# ATTRS2 = ATTRS_VANILLA
ATTRS1 = rtm_paras.keys()
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# ATTRS1 = ['LAI']
# ATTRS2 = ['cab', 'cw', 'LAIu']
df = df3
df_coniferous = df[df['class'].isin(coniferous)]
df_deciduous = df[df['class'].isin(deciduous)]
for i, attr1 in enumerate(ATTRS1):
    # drop the attr in ATTRS2 that is the same as attr1
    ATTRS2 = [attr for attr in rtm_paras.keys() if attr != attr1]
    for j, attr2 in enumerate(ATTRS2):
        ax = axs[i, j]
        # ax = axs[j]
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_coniferous, ax=ax, s=8, alpha=0.5, color='red')
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_deciduous, ax=ax, s=8, alpha=0.5, color='blue')
        fontsize = 30
        # set the distance between the title and the plot
        # ax.set_title(f'{attr1} v. {attr2}', fontsize=fontsize, pad=10)
        ax.set_xlabel(ATTRS_LATEX[attr1], fontsize=fontsize)
        ax.set_xlim(rtm_paras[attr1]['min'], rtm_paras[attr1]['max'])
        ax.set_ylabel(ATTRS_LATEX[attr2], fontsize=fontsize)
        ax.set_ylim(rtm_paras[attr2]['min'], rtm_paras[attr2]['max'])
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=25)
        # ax.legend(fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'scatter_realset_vars_pairs_nn_coniferous_v_deciduous.png'))
# plt.savefig(os.path.join(
#     SAVE_PATH, 'scatter_realset_vars_pairs_nn_coniferous_v_deciduous_3_LAI.png'))
plt.show()

#%% SKIP NEW scatter plot for bias pairs for AE_RTM_corr
fig, axs = plt.subplots(11, 11, figsize=(35, 35))
for i, band1 in enumerate(S2_BANDS):
    for j, band2 in enumerate(S2_BANDS):
        ax = axs[i, j]
        sns.scatterplot(x=f'bias_{band1}', y=f'bias_{band2}',
                        data=df2, ax=ax, s=8, alpha=0.5)
        fontsize = 22
        # set the distance between the title and the plot
        ax.set_xlabel(band1, fontsize=fontsize)
        ax.set_ylabel(band2, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'scatter_realset_bias_pairs_corr.png'))

# %% NEW scatter plot for bias pairs distinguishing Coniferous and Diceduous for AE_RTM_corr
fig, axs = plt.subplots(11, 11, figsize=(35, 35))
df_coniferous = df2[df2['class'].isin(coniferous)]
df_deciduous = df2[df2['class'].isin(deciduous)]
for i, band1 in enumerate(S2_BANDS):
    for j, band2 in enumerate(S2_BANDS):
        ax = axs[i, j]
        sns.scatterplot(x=f'bias_{band1}', y=f'bias_{band2}',
                        data=df_coniferous, ax=ax, s=8, alpha=0.5, color='red')
        sns.scatterplot(x=f'bias_{band1}', y=f'bias_{band2}',
                        data=df_deciduous, ax=ax, s=8, alpha=0.5, color='blue')
        fontsize = 22
        # set the distance between the title and the plot
        ax.set_xlabel(band1, fontsize=fontsize)
        ax.set_ylabel(band2, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'scatter_realset_bias_pairs_corr_coniferous_v_deciduous.png'))
# %%
"""
Scatter plot of input and reconstruction bands
Plot for VanillaAE, AE_RTM, AE_RTM_corr NOTE neurips 2 bands compared
TODO until here 2024.05.21 tmw resume from here. copy and paste and create a new plot
"""
# Scatter plot of the input and reconstructed bands
# fig, axs = plt.subplots(3, 4, figsize=(24, 16))
fig, axs = plt.subplots(1, 4, figsize=(24, 5))
# S2_BANDS = ['B02_BLUE', 'B05_RE1', 'B08_NIR1', 'B11_SWI1']
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# S2_BANDS = ['B02_BLUE', 'B08_NIR1']
# df = df2
# for i, band in enumerate(S2_BANDS):
i = 0
for band in ['B02_BLUE', 'B08_NIR1']:
    for df in [df2, df1]:
        if i%2 == 0:
            color = 'blue'
            label = 'w/ $\mathbf{C}$'
        else:
            color = 'red'
            label = 'w/o $\mathbf{C}$'
        # ax = axs[i//4, i % 4]
        ax = axs[i]
        i+=1
        sns.scatterplot(x='target_'+band, y='output_'+band, data=df, ax=ax,
                        s=8, alpha=0.5, color=color, label=label)
        # adjust the point size and alpha and color
        # calculate RMSE and add it to the title
        # rmse = np.sqrt(np.mean((df[f'target_{band}'] - df[f'output_{band}'])**2))
        r2 = r_square(df[f'target_{band}'], df[f'output_{band}'])
        fontsize = 30
        # add the RMSE to the title
        ax.set_title(S2_names[band], fontsize=fontsize)
        xlabel = '$X_{\mathrm{S2}}$'
        ylabel = '$X_{\mathrm{S2, B}}$' if i%2 == 0 else '$X_{\mathrm{S2, C}}$'
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=25)
        # plot the diagonal line
        limits = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # round the limit ticks to 2 decimal places
        # limits = [np.round(limits[0], 2), np.round(limits[1], 2)]
        ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        # set the distance between y label and y axis
        ax.yaxis.labelpad = 10
        ax.set_aspect('equal')
        # make sure both axes have same ticks to display
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        # make sure all ticks are rounded to 2 decimal places
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        # set RMSE as a legend
        # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
        ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
# make the last subplot empty
# axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_realset_bands_corr_v_wocorr_target_v_output.png'))
plt.show()

# %%
"""
Scatter plot of input and reconstruction bands
Plot for VanillaAE, AE_RTM, AE_RTM_corr NOTE neurips full plots
"""
# Scatter plot of the input and reconstructed bands
fig, axs = plt.subplots(3, 4, figsize=(24, 16))
df = df3
# color = 'blue'
color = 'red'
ylabel = '$X_{\mathrm{S2, D}}$' 
for i, band in enumerate(S2_BANDS):
    ax = axs[i//4, i % 4]
    sns.scatterplot(x='target_'+band, y='output_'+band, data=df, ax=ax,
                    s=8, alpha=0.5, color=color, label=label)
    # adjust the point size and alpha and color
    # calculate RMSE and add it to the title
    # rmse = np.sqrt(np.mean((df[f'target_{band}'] - df[f'output_{band}'])**2))
    r2 = r_square(df[f'target_{band}'], df[f'output_{band}'])
    fontsize = 30
    # add the RMSE to the title
    ax.set_title(S2_names[band], fontsize=fontsize)
    xlabel = '$X_{\mathrm{S2}}$'
    # ylabel = '$X_{\mathrm{S2, C}}$' 
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # set the same ticks for both x and y axes
    ax.tick_params(axis='both', which='major', labelsize=25)
    # plot the diagonal line
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # round the limit ticks to 2 decimal places
    # limits = [np.round(limits[0], 2), np.round(limits[1], 2)]
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    # set the distance between y label and y axis
    ax.yaxis.labelpad = 10
    ax.set_aspect('equal')
    # make sure both axes have same ticks to display
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
    # make sure all ticks are rounded to 2 decimal places
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    # set RMSE as a legend
    # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
    ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
# make the last subplot empty
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_realset_bands_nn_target_v_output.png'))
plt.show()

# %% NOTE neurips changed
"""
For each variable, cluster the sample according to forest type and then
according to the date of acquisition. Get the mean and std of the clustered
samples and then plot the time series of the mean and show the std as error bars.
Plot for VanillaAE, AE_RTM, AE_RTM_corr, NNRegressor
"""
# ATTRS2 = ATTRS_VANILLA
ATTRS2 = rtm_paras.keys()
# ATTRS2 = ['N', 'cab', 'LAIu', 'fc']
# ATTRS2 = ['LAIu', 'fc']
# fig, axs = plt.subplots(2, 4, figsize=(35, 12)) 
fig, axs = plt.subplots(4, 2, figsize=(20, 20)) 
# fig, axs = plt.subplots(1, 4, figsize=(30, 5))
# fig, axs = plt.subplots(1, 3, figsize=(30, 5))
df = df3
for i, attr in enumerate(ATTRS2):
# for i, attr in enumerate(['LAI', 'LAIu', 'fc']):
    # ax = axs[i//4, i % 4]
    ax = axs[i//2, i % 2]
    # ax = axs[i]
    # get the time seris of mean and std of the clustered samples for each variable
    mean_coniferous = []
    std_coniferous = []
    mean_deciduous = []
    std_deciduous = []
    for date in dates:
        df_filtered = df[df['date']==date]
        mean_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].mean())
        std_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].std())
        mean_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].mean())
        std_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].std())
    
    # plot the time series of the mean and show the std as error bars
    # map each date to the format like Aug 21, Apr 08, etc.
    dates_plot = []
    months = {'04': 'Apr', '05': 'May', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct'}
    dates_plot = [f"{date.split('.')[2]} {months[date.split('.')[1]]}" for date in dates]
    # df is df3, plot it in the background with a lighter color than df2
    # dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
    ax.errorbar(x=dates_plot, y=mean_coniferous, yerr=std_coniferous, fmt='o', color = 'red', label='Coniferous')
    ax.errorbar(x=dates_plot, y=mean_deciduous, yerr=std_deciduous, fmt='o', color = 'blue', label='Deciduous')
    fontsize = 32
    # ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel(ATTRS_LATEX[attr], fontsize=fontsize)
    # set the range of y axis as the physical range of the variable
    ax.set_ylim(rtm_paras[attr]['min'], rtm_paras[attr]['max'])
    ax.legend(fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=23)
    # rotate the ticks
    ax.set_xticklabels(dates_plot, rotation=-45)  

axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_vars_nn_coniferous_v_deciduous.png'))
plt.show()

# %% NEW randomly select five samples and plot the time series of the mean and show the std as error bars for AE_RTM_corr
# randomly select five samples
# TODO show the spectral signature (original prediction and corrected) of individual samples
# or per species. Also according to the forest type.
ATTRS = rtm_paras.keys()
sample_ids = np.random.choice(df2['sample_id'].unique(), 5)
for sample_id in sample_ids:
    fig, axs = plt.subplots(2, 4, figsize=(30, 10))
    for i, attr in enumerate(ATTRS):
        ax = axs[i//4, i % 4]
        pts = []
        # get the time seris of mean and std of the clustered samples for each variable
        for date in dates:
            df_filtered = df2[df2['date']==date]
            pts.append(df_filtered[df_filtered['sample_id']==sample_id][f'latent_{attr}'].values[0])
        dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
        ax.plot(dates_plot, pts, 'o-', color='blue')
        fontsize = 18
        # set the distance between the title and the plot
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(attr, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=10)
    axs[-1, -1].axis('off')
    sample_forest = 'Coniferous' if df2[df2['sample_id']==sample_id]['class'].values[0] in coniferous else 'Deciduous'
    sample_species = df2[df2['sample_id']==sample_id]['class'].values[0]
    # Set the title of the figure
    plt.suptitle(f"{sample_forest}: {sample_species}", fontsize=22)
    # Set the distance between the title and the plot
    plt.subplots_adjust(top=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'timeseries_realset_vars_corr_{sample_forest}_{sample_species}_{sample_id}.png'))
#%% Derieve the mean and std of each estimated variables for each species 
# the row would be the 12 species and the columns would be the 7 variables
# Do it for NNRegressor, AE_RTM and AE_RTM_corr
# Save the data in a format that can be copied to a latex table   
# each row in the table: species & mean1 \pm std1 & mean2 \pm std2 & ... & mean7 \pm std7 \\
# write the data in a text file, which can be used to generate a latex table
ATTRS = rtm_paras.keys()
df = df3
# create a csv file to write the data
with open(os.path.join(SAVE_PATH, 'mean_std_realset_vars_nn.txt'), 'w') as f:
    f.write('Species & N & cab & cw & cm & LAI & LAIu & fc \\\\ \n')
    for species in coniferous+deciduous:
        df_filtered = df[df['class']==species]
        mean = []
        std = []
        for attr in ATTRS:
            # if attr in ['cw', 'cm'], scale the mean and std by 100
            if attr in ['cw', 'cm']:
                mean.append(df_filtered[f'latent_{attr}'].mean()*100)
                std.append(df_filtered[f'latent_{attr}'].std()*100)
            else:
                mean.append(df_filtered[f'latent_{attr}'].mean())
                std.append(df_filtered[f'latent_{attr}'].std())
        f.write(f"{species} & {mean[0]:.2f} $\\pm$ {std[0]:.2f} & {mean[1]:.2f} $\\pm$ {std[1]:.2f} & {mean[2]:.2f} $\\pm$ {std[2]:.2f} & {mean[3]:.2f} $\\pm$ {std[3]:.2f} & {mean[4]:.2f} $\\pm$ {std[4]:.2f} & {mean[5]:.2f} $\\pm$ {std[5]:.2f} & {mean[6]:.2f} $\\pm$ {std[6]:.2f} \\\\ \n")
#%% 
"""
Calculate the Jeffreys-Matusita (JM) distance for species pairs based on the estimated variables of each species
JM distance is a measure of the similarity between two distributions
The JM distance is calculated as follows:
JM = sqrt(2*(1 - exp(-D^2/2)))
where D is the Bhattacharyya distance
D = -ln(sum(sqrt(p(x)*q(x))))
where p(x) and q(x) are the probability density functions of the two distributions
"""
# Assuming df2 is your DataFrame, rtm_paras holds parameters, and SAVE_PATH is defined
ATTRS = list(rtm_paras.keys())
df = df2

species_list = coniferous + deciduous  # Combine your species lists
means = {}  # Dictionary to store mean vectors
covariances = {}  # Dictionary to store covariance matrices

# Calculate means and covariances
for species in species_list:
    df_filtered = df[df['class'] == species]
    mean = df_filtered[[f'latent_{attr}' for attr in ATTRS]].mean().to_numpy()
    covariance = df_filtered[[f'latent_{attr}' for attr in ATTRS]].cov().to_numpy()
    means[species] = mean
    covariances[species] = covariance

# Bhattacharyya distance function
def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    cov_mean = (cov1 + cov2) / 2
    term1 = 0.125 * np.dot(np.dot((mean1 - mean2).T, np.linalg.inv(cov_mean)), (mean1 - mean2))
    term2 = 0.5 * np.log(np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    return term1 + term2

# Jeffreys-Matusita distance function
def jeffreys_matusita_distance(b_distance):
    return 2 * (1 - np.exp(-b_distance))

# Calculating pairwise distances
jm_distances = pd.DataFrame(np.zeros((len(species_list), len(species_list))), index=species_list, columns=species_list)

for i, species1 in enumerate(species_list):
    for j, species2 in enumerate(species_list[i+1:], start=i+1):
        b_distance = bhattacharyya_distance(means[species1], covariances[species1], means[species2], covariances[species2])
        jm_distance = jeffreys_matusita_distance(b_distance)
        jm_distances.at[species1, species2] = jm_distance
        jm_distances.at[species2, species1] = jm_distance
# Now jm_distances contains the pairwise Jeffreys-Matusita distances
# You can save this to a file, or directly print it
# print(jm_distances)
plt.figure(figsize=(12.5, 11))
# ax = sns.heatmap(jm_distances, annot=True, cmap='viridis', vmin=0, vmax=2, 
#             annot_kws={"size": 16})
# plot without the annotation 
ax = sns.heatmap(jm_distances, cmap='viridis', vmin=0, vmax=2)  
# highlight the blocks with the same species 
rect1 = patches.Rectangle((0, 0), 5, 5, linewidth=3, edgecolor='red', facecolor='none')
rect2 = patches.Rectangle((5, 5), 7, 7, linewidth=3, edgecolor='blue', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
# set the size of ticks and colorbar
fontsize = 19
plt.tick_params(axis='both', which='major', labelsize=fontsize)
# set the label size of the colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fontsize)
# color the first five ticks in red and the last seven in blue
for i, label in enumerate(ax.get_yticklabels()):
    if i < 5:
        label.set_color('red')
    else:
        label.set_color('blue')
for i, label in enumerate(ax.get_xticklabels()):
    if i < 5:
        label.set_color('red')
    else:
        label.set_color('blue')
# save the heatmap
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'heatmap_jm_distance_realset_species_corr.png'), dpi=300)
plt.show()
# %% NEW plot the time series of the mean and show the std as error bars for AE_RTM_corr
df = df2
fig, axs = plt.subplots(3, 4, figsize=(35, 18))
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
for i, band in enumerate(S2_BANDS):
    ax = axs[i//4, i % 4]
    # get the time seris of mean and std of the clustered samples for each variable
    mean_coniferous = []
    std_coniferous = []
    mean_deciduous = []
    std_deciduous = []
    for date in dates:
        df_filtered = df[df['date']==date]
        mean_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'bias_{band}'].mean())
        std_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'bias_{band}'].std())
        mean_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'bias_{band}'].mean())
        std_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'bias_{band}'].std())
    
    # plot the time series of the mean and show the std as error bars
    # Plot only the month and day of the date (original format 'yyyy.mm.dd', to this formar like dd/mm)
    # dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
    # map each date to the format like Aug 21, Apr 08, etc.
    dates_plot = []
    months = {'04': 'Apr', '05': 'May', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct'}
    dates_plot = [f"{date.split('.')[2]} {months[date.split('.')[1]]}" for date in dates]
    ax.errorbar(x=dates_plot, y=mean_coniferous, yerr=std_coniferous, fmt='o', color = 'red', label='Coniferous')
    ax.errorbar(x=dates_plot, y=mean_deciduous, yerr=std_deciduous, fmt='o', color = 'blue', label='Deciduous')
    fontsize = 30
    # ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel('Bias', fontsize=fontsize)
    ax.set_title(S2_names[band], fontsize=fontsize)
    # set 0 as the middle of the y axis by getting the maximum absolute value of the bias
    ax.set_ylim(-np.max(np.abs(df[f'bias_{band}'])), np.max(np.abs(df[f'bias_{band}'])))
    ax.legend(fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=25)
    # rotate the ticks
    ax.set_xticklabels(dates_plot, rotation=-45)
axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_bias_corr_coniferous_v_deciduous.png'))

# %%  NOTE neurips made changes
# Plot the spectral signature of the input and output bands for AE_RTM_corr with mean and std as error bars
# first plot for coniferous and deciduous forests
# then for each species
# and then for five randomly selected samples
# Plot a single figure for each forest type
S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
df = df2
# fig, axs = plt.subplots(1, 2, figsize=(22, 5))
# each subplot for comparing the spectral signature of the input and output bands for a single forest type
for i, forest_type in enumerate([coniferous, deciduous]):
    forest = 'Coniferous' if i==0 else 'Deciduous'
    plt.figure(figsize=(10, 4))
    mean_input = []
    std_input = []
    mean_output = []
    std_output = []
    for band in S2_BANDS:
        mean_input.append(df[df['class'].isin(forest_type)][f'init_output_{band}'].mean())
        std_input.append(df[df['class'].isin(forest_type)][f'init_output_{band}'].std())
        mean_output.append(df[df['class'].isin(forest_type)][f'output_{band}'].mean())
        std_output.append(df[df['class'].isin(forest_type)][f'output_{band}'].std())
    plt.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_input, yerr=std_input, fmt='o', color = 'red', label='RTM Output')
    plt.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Output')
    fontsize = 25
    plt.xlabel('Bands', fontsize=fontsize)
    plt.ylabel('Reflectance', fontsize=fontsize)
    # set the limit of y axis
    plt.ylim(0, 0.5)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'realset_spectral_signature_corr_init_v_final_{forest}.png'))
   
# %% Spectral signature per species
df = df2
# fig, axs = plt.subplots(6, 2, figsize=(22, 30))
fig, axs = plt.subplots(4, 3, figsize=(33, 20))
# each subplot for comparing the spectral signature of the input and output bands for a single species
for i, species in enumerate(coniferous+deciduous):
    # ax = axs[i//2, i % 2]
    ax = axs[i//3, i % 3]
    # get the time seris of mean and std of the clustered samples for each variable
    mean_input = []
    std_input = []
    mean_output = []
    std_output = []
    for band in S2_BANDS:
        mean_input.append(df[df['class']==species][f'init_output_{band}'].mean())
        std_input.append(df[df['class']==species][f'init_output_{band}'].std())
        mean_output.append(df[df['class']==species][f'output_{band}'].mean())
        std_output.append(df[df['class']==species][f'output_{band}'].std())
    # plot the time series of the mean and show the std as error bars
    ax.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_input, yerr=std_input, fmt='o', color = 'red', label='RTM Output')
    ax.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Output')
    fontsize = 30
    forest_type = 'Coniferous' if species in coniferous else 'Deciduous'
    ax.set_title(f'{forest_type}: {species}', fontsize=fontsize, pad=10)
    ax.set_xlabel('Bands', fontsize=fontsize)
    ax.set_ylabel('Reflectance', fontsize=fontsize)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'realset_spectral_signature_corr_init_v_final_species.png'))

# %% Spectral signature for five randomly selected samples
# randomly select 2 date for each slected samples
# df = df2
df = df2[df2['class'].isin(coniferous)]
sample_ids = np.random.choice(df['sample_id'].unique(), 5)
sample_dates = np.random.choice(df['date'].unique(), 1)
for sample_id in sample_ids:
    for sample_date in sample_dates:
        df_filtered = df[(df['sample_id']==sample_id) & (df['date']==sample_date)]
        # df_filtered is a single sample
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_input = []
        std_input = []
        mean_output = []
        std_output = []
        for band in S2_BANDS:
            mean_input.append(df_filtered[f'init_output_{band}'].values[0])
            std_input.append(0)
            mean_output.append(df_filtered[f'output_{band}'].values[0])
            std_output.append(0)
        ax.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_input, yerr=std_input, fmt='o', color = 'red', label='Initial Spectra')
        ax.errorbar(x=[S2_names[band] for band in S2_BANDS], y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Spectra')
        fontsize = 25
        forest_type = 'Coniferous' if df_filtered['class'].values[0] in coniferous else 'Deciduous'
        species = df_filtered['class'].values[0]
        ax.set_title(f'{forest_type}: {species}', fontsize=fontsize, pad=10)
        ax.set_xlabel('Bands', fontsize=fontsize)
        ax.set_ylabel('Spectral Signature', fontsize=fontsize)
        ax.set_ylim(0, 0.5)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(
            SAVE_PATH, 'realset_spectral_signature_individual_sample', 
            f'realset_spectral_signature_corr_init_{forest_type}_{species}_{sample_id}_{sample_date}.png'))
        plt.show()
    
# %% Plot a R-squared heatmap between variables learned by VanillaAE and AE_RTM_corr
ATTRS1 = ATTRS_VANILLA[1:]
ATTRS2 = ATTRS
# derive one matrix of R-squared values and plot it as a heatmap
mat = np.zeros((len(ATTRS1), len(ATTRS2)))
fig, ax = plt.subplots(figsize=(10, 8))
for i, attr1 in enumerate(ATTRS1):
    for j, attr2 in enumerate(ATTRS2):
        # calculating the R-squared value between two variables using equations
        mat[i, j] = np.corrcoef(df0[f'latent_{attr1}'].values, df2[f'latent_{attr2}'].values)[0, 1]
# plot the heatmap
sns.heatmap(mat, annot=True, ax=ax, cmap='Blues', vmin=-1, 
            vmax=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 16})
# match the size of the scale bar as the size of the heatmap
ax.set_aspect('equal')
ax.set_xticklabels(ATTRS2, fontsize=16)
ax.set_yticklabels(ATTRS1, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'heatmap_correlaton_realset_vars_vanilla_v_corr.png'))

# %% plot the histogram of the six latent variables

