# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    BASE_PATH, 'AE_RTM_corr/0124_000330/model_best_testset_analyzer.csv')
CSV_PATH3 = os.path.join(
    BASE_PATH, 'NNRegressor/0124_160519/model_best_testset_analyzer_real.csv')

# NOTE models trained with embedded conditions on the same data
# CSV_PATH1 = os.path.join(
#     BASE_PATH, 'AE_RTM_con/0201_140847/model_best_testset_analyzer.csv')
# CSV_PATH2 = os.path.join(
#     BASE_PATH, 'AE_RTM_corr_con/0201_201257/model_best_testset_analyzer.csv')

SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr/0124_000330/plots')
# SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr_con/0201_201257/plots')

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
rtm_paras = json.load(open('/maps/ys611/ai-refined-rtm/configs/rtm_paras.json'))
ATTRS = list(rtm_paras.keys())
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


# %%
"""
Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
N2, NNRegressor and AE_RTM_corr
"""
# Histogram of the latent variables of both models (AE_RTM and AE_RTM_corr)
NUM_BINS = 100
# ATTRS = ['N', 'cab', 'LAIu', 'fc']
ATTRS = ['N', 'fc']
# create one figure and plot both variable predictions of different models as a subplot
# fig, axs = plt.subplots(2,4, figsize = (25, 10))
# fig, axs = plt.subplots(1,4, figsize = (25, 5))
fig, axs = plt.subplots(1, 2, figsize = (12.5, 5))
for i, attr in enumerate(ATTRS):
    # ax=axs[i//4, i % 4]
    ax = axs[i]
    # sns.histplot(
    #     df1[f'latent_{attr}'].values,
    #     bins=NUM_BINS,
    #     ax=ax,
    #     color='red',
    #     label='AE_RTM',
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
    #     label='AE_RTM_corr',
    #     alpha=0.6,
    # )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=22)
    fontsize = 30
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.legend(fontsize=22)
# remove the last subplot
# axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_NN_2.png'), dpi=300)
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
# ATTRS = ['N', 'cab', 'LAIu', 'fc']
ATTRS = ['N', 'fc']
# create one figure and plot both variable predictions of different models as a subplot
# fig, axs = plt.subplots(2, 4, figsize=(25, 10))
# fig, axs = plt.subplots(1, 4, figsize=(25, 5))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
for i, attr in enumerate(ATTRS):
    # ax=axs[i//4, i % 4]
    ax = axs[i]
    sns.histplot(
        df_coniferous[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        label='Coniferous',
        alpha=0.6,
    )
    sns.histplot(
        df_deciduous[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        # label='AE_RTM_syn',
        label='Deciduous',
        alpha=0.6,
    )
    # change the fontsize of the x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=22)
    fontsize = 30
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
    ax.legend(fontsize=22)
# axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_corr_coniferous_v_deciduous_2.png'), dpi=300)
plt.show()
    
# %%
"""
Histogram of the latent variables per tree species for AE_RTM_corr
"""
# Histogram of the latent variables for selected species 
# Plot histogram of biophysical variables of each single species
# TODO plot the spectral signature for each species before and after correction
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
        ax.tick_params(axis='both', which='major', labelsize=16)
        fontsize = 20
        ax.set_xlabel(attr, fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
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
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(band, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
axs[-1, -1].axis('off')
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, 'histogram_realset_bias_corr.png'), dpi=300)
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
    fontsize = 20
    # set the distance between the title and the plot
    # ax.set_title(f'{var_pair[0]} v. {var_pair[1]}', fontsize=fontsize, pad=10)
    ax.set_xlabel(var_pair[0], fontsize=fontsize)
    ax.set_ylabel(var_pair[1], fontsize=fontsize)
    # set the same ticks for both x and y axes
    ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'scatter_realset_vars_corr_fc_v_cd_h.png'))

# %%
"""
Scatter plot for variable pairs for VanillaAE, AE_RTM, AE_RTM_corr, and NNRegressor
"""
fig, axs = plt.sublots(7, 7, figsize=(25, 25))
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
        ax.set_xlabel(attr1, fontsize=fontsize)
        ax.set_ylabel(attr2, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'scatter_realset_vars_pairs_corr.png'))
# %%
"""
Scatter plot for variable pairs distinguishing Coniferous and Diceduous 
for VanillaAE, and AE_RTM_corr
"""
# fig, axs = plt.subplots(7, 7, figsize=(25, 25))
# drop the last two elements in ATTRS (cd and h)
# ATTRS2 = ATTRS_VANILLA
# ATTRS2 = rtm_paras.keys()
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
ATTRS1 = ['LAI']
ATTRS2 = ['cab', 'cw', 'LAIu']
df = df3
df_coniferous = df[df['class'].isin(coniferous)]
df_deciduous = df[df['class'].isin(deciduous)]
for i, attr1 in enumerate(ATTRS1):
    for j, attr2 in enumerate(ATTRS2):
        # ax = axs[i, j]
        ax = axs[j]
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_coniferous, ax=ax, s=8, alpha=0.5, color='red')
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_deciduous, ax=ax, s=8, alpha=0.5, color='blue')
        fontsize = 30
        # set the distance between the title and the plot
        # ax.set_title(f'{attr1} v. {attr2}', fontsize=fontsize, pad=10)
        # make the x label starting from 0
        ax.set_xlim(left=0)
        ax.set_xlabel(attr1, fontsize=fontsize)
        ax.set_ylabel(attr2, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=22)
        # ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'scatter_realset_vars_pairs_nn_coniferous_v_deciduous_6.png'))

#%% NEW scatter plot for bias pairs for AE_RTM_corr
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
Plot for VanillaAE, AE_RTM, AE_RTM_corr
"""
# Scatter plot of the input and reconstructed bands
# fig, axs = plt.subplots(3, 4, figsize=(20, 15))
# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# S2_BANDS = ['B02_BLUE', 'B05_RE1', 'B08_NIR1', 'B11_SWI1']
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
S2_BANDS = ['B02_BLUE', 'B08_NIR1']
df = df3
for i, band in enumerate(S2_BANDS):
    # ax = axs[i//4, i % 4]
    ax = axs[i]
    sns.scatterplot(x='target_'+band, y='output_'+band,
                    data=df, ax=ax, s=8, alpha=0.5)
    # adjust the point size and alpha and color
    # calculate RMSE and add it to the title
    rmse = np.sqrt(np.mean((df[f'target_{band}'] - df[f'output_{band}'])**2))
    fontsize = 18
    # add the RMSE to the title
    ax.set_title(f'{band.split("_")[0]}: RMSE={rmse:.3f}', fontsize=fontsize)
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
# axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_realset_bands_nn_target_v_output_2.png'))
plt.show()

# %%
"""
For each variable, cluster the sample according to forest type and then
according to the date of acquisition. Get the mean and std of the clustered
samples and then plot the time series of the mean and show the std as error bars.
Plot for VanillaAE, AE_RTM, AE_RTM_corr, NNRegressor
"""
# ATTRS2 = ATTRS_VANILLA
# ATTRS2 = ATTRS
# ATTRS2 = ['N', 'cab', 'LAIu', 'fc']
ATTRS2 = ['LAIu', 'fc']
# fig, axs = plt.subplots(2, 4, figsize=(30, 10)) 
# fig, axs = plt.subplots(1, 4, figsize=(30, 5))
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for i, attr in enumerate(ATTRS2):
    # ax = axs[i//4, i % 4]
    ax = axs[i]
    # get the time seris of mean and std of the clustered samples for each variable
    mean_coniferous = []
    std_coniferous = []
    mean_deciduous = []
    std_deciduous = []
    for date in dates:
        df_filtered = df2[df2['date']==date]
        mean_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].mean())
        std_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].std())
        mean_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].mean())
        std_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].std())
    
    # plot the time series of the mean and show the std as error bars
    # map each date to the format like Aug 21, Apr 08, etc.
    dates_plot = []
    months = {'04': 'Apr', '05': 'May', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct'}
    dates_plot = [f"{date.split('.')[2]} {months[date.split('.')[1]]}" for date in dates]
    # dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
    ax.errorbar(x=dates_plot, y=mean_coniferous, yerr=std_coniferous, fmt='o', color = 'red', label='Coniferous')
    ax.errorbar(x=dates_plot, y=mean_deciduous, yerr=std_deciduous, fmt='o', color = 'blue', label='Deciduous')
    fontsize = 30
    ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel(attr, fontsize=fontsize)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=21)
    # rotate the ticks
    ax.set_xticklabels(dates_plot, rotation=-45)
# axs[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_vars_corr_coniferous_v_deciduous_2.png'))
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
df = df2
# create a csv file to write the data
with open(os.path.join(SAVE_PATH, 'mean_std_realset_vars_corr.txt'), 'w') as f:
    f.write('Species & N & cab & cw & cm & LAI & LAIu & fc \\\\ \n')
    for species in coniferous+deciduous:
        df_filtered = df[df['class']==species]
        mean = []
        std = []
        for attr in ATTRS:
            mean.append(df_filtered[f'latent_{attr}'].mean())
            std.append(df_filtered[f'latent_{attr}'].std())
        f.write(f"{species} & {mean[0]:.2f} $\\pm$ {std[0]:.2f} & {mean[1]:.2f} $\\pm$ {std[1]:.2f} & {mean[2]:.2f} $\\pm$ {std[2]:.2f} & {mean[3]:.2f} $\\pm$ {std[3]:.2f} & {mean[4]:.2f} $\\pm$ {std[4]:.2f} & {mean[5]:.2f} $\\pm$ {std[5]:.2f} & {mean[6]:.2f} $\\pm$ {std[6]:.2f} \\\\ \n")


# %% NEW plot the time series of the mean and show the std as error bars for AE_RTM_corr
fig, axs = plt.subplots(3, 4, figsize=(28, 15))
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
        df_filtered = df2[df2['date']==date]
        mean_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'bias_{band}'].mean())
        std_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'bias_{band}'].std())
        mean_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'bias_{band}'].mean())
        std_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'bias_{band}'].std())
    
    # plot the time series of the mean and show the std as error bars
    # Plot only the month and day of the date (original format 'yyyy.mm.dd', to this formar like dd/mm)
    dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
    ax.errorbar(x=dates_plot, y=mean_coniferous, yerr=std_coniferous, fmt='o', color = 'red', label='Coniferous')
    ax.errorbar(x=dates_plot, y=mean_deciduous, yerr=std_deciduous, fmt='o', color = 'blue', label='Deciduous')
    fontsize = 18
    ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel(band, fontsize=fontsize)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=10)
axs[-1, -1].axis('off')
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_bias_corr_coniferous_v_deciduous.png'))

# %% 
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
    plt.figure(figsize=(10, 5))
    mean_input = []
    std_input = []
    mean_output = []
    std_output = []
    for band in S2_BANDS:
        mean_input.append(df[df['class'].isin(forest_type)][f'init_output_{band}'].mean())
        std_input.append(df[df['class'].isin(forest_type)][f'init_output_{band}'].std())
        mean_output.append(df[df['class'].isin(forest_type)][f'output_{band}'].mean())
        std_output.append(df[df['class'].isin(forest_type)][f'output_{band}'].std())
    # plt.errorbar(x=S2_BANDS, y=mean_input, yerr=std_input, fmt='o', color = 'red', label='Initial Output')
    plt.errorbar(x=S2_BANDS, y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Output')
    fontsize = 18
    plt.xlabel('Bands', fontsize=fontsize)
    plt.ylabel('Spectral Signature', fontsize=fontsize)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    forest = 'Coniferous' if i==0 else 'Deciduous'
    plt.savefig(os.path.join(SAVE_PATH, f'realset_spectral_signature_corr_output_{forest}.png'))
   
# %% Spectral signature per species
df = df2
fig, axs = plt.subplots(6, 2, figsize=(22, 30))
# each subplot for comparing the spectral signature of the input and output bands for a single species
for i, species in enumerate(coniferous+deciduous):
    ax = axs[i//2, i % 2]
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
    # ax.errorbar(x=S2_BANDS, y=mean_input, yerr=std_input, fmt='o', color = 'red', label='Initial Output')
    ax.errorbar(x=S2_BANDS, y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Output')
    fontsize = 18
    forest_type = 'Coniferous' if species in coniferous else 'Deciduous'
    ax.set_title(f'{forest_type}: {species}', fontsize=fontsize, pad=10)
    ax.set_xlabel('Bands', fontsize=fontsize)
    ax.set_ylabel('Spectral Signature', fontsize=fontsize)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'realset_spectral_signature_corr_output_species.png'))

# %% Spectral signature for five randomly selected samples
# randomly select 2 date for each slected samples
df = df2
# df = df2[df2['class'].isin(deciduous)]
sample_ids = np.random.choice(df['sample_id'].unique(), 5)
sample_dates = np.random.choice(df['date'].unique(), 2)
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
        ax.errorbar(x=S2_BANDS, y=mean_input, yerr=std_input, fmt='o', color = 'red', label='Initial Output')
        ax.errorbar(x=S2_BANDS, y=mean_output, yerr=std_output, fmt='o', color = 'blue', label='Corrected Output')
        fontsize = 18
        forest_type = 'Coniferous' if df_filtered['class'].values[0] in coniferous else 'Deciduous'
        species = df_filtered['class'].values[0]
        ax.set_title(f'{forest_type}: {species}', fontsize=fontsize, pad=10)
        ax.set_xlabel('Bands', fontsize=fontsize)
        ax.set_ylabel('Spectral Signature', fontsize=fontsize)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(
            SAVE_PATH, 'realset_spectral_signature_individual_sample', 
            f'realset_spectral_signature_corr_init_{forest_type}_{species}_{sample_id}_{sample_date}.png'))
    
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

# %%
