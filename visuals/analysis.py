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
CSV_PATH1 = os.path.join(
    BASE_PATH, 'AE_RTM/0123_175312/model_best_testset_analyzer.csv')
CSV_PATH2 = os.path.join(
    BASE_PATH, 'AE_RTM_corr/0124_000330/model_best_testset_analyzer.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'AE_RTM_corr/0124_000330/plots')
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
df2[[f'bias_{band}' for band in S2_BANDS]] = df2[[f'bias_{band}' for band in S2_BANDS]]*SCALE + MEAN

# get the unique tree species
coniferous = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra', 
              'Larix decidua', 'Pinus sylvestris']
deciduous = ['Prunus spp', 'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp', 
             'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']
df_coniferous = df2[df2['class'].isin(coniferous)]
df_deciduous = df2[df2['class'].isin(deciduous)]
dates = ['2018.04.08', '2018.04.21', '2018.05.06', '2018.07.02', '2018.08.09', 
         '2018.08.21', '2018.08.29', '2018.09.13', '2018.09.18', '2018.09.28', 
         '2018.09.30', '2018.10.05', '2018.10.10', '2018.10.30']


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
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_realset_vars_AE_RTM_v_corr.png'), dpi=300)
plt.show()

# %%
"""
Histogram of the latent variables between Coniferous and Deciduous for AE_RTM_corr
"""
# TODO finish this code
# Histogram of the latent variables of selected species
NUM_BINS = 100
# create one figure and plot both variable predictions of different models as a subplot
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
for i, attr in enumerate(ATTRS):
    ax=axs[i//3, i % 3]
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
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
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
    # Set the title of the figure
    forest_type = 'Coniferous' if species in coniferous else 'Deciduous'
    plt.suptitle(f"{forest_type}: {species}", fontsize=22)
    # Set the distance between the title and the plot
    plt.subplots_adjust(top=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'histogram_realset_vars_corr_{species}.png'), dpi=300)
    plt.show()

# %%
"""
Scatter plot for selected variables for AE_RTM_corr
"""    
# subplots for fc v. cd, fc v. h
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
var_pairs = [['fc', 'cd'], ['fc', 'h']]
for i, var_pair in enumerate(var_pairs):
    ax = axs[i]
    sns.scatterplot(x=f'latent_{var_pair[0]}', y=f'latent_{var_pair[1]}',
                    data=df2, ax=ax, s=8, alpha=0.5)
    fontsize = 18
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
Scatter plot for variable pairs for AE_RTM_corr
"""
fig, axs = plt.subplots(7, 7, figsize=(25, 25))
# drop the last two elements in ATTRS (cd and h)
ATTRS2 = ATTRS[:-2]
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
Scatter plot for variable pairs distinguishing Coniferous and Diceduous for AE_RTM_corr
"""
fig, axs = plt.subplots(7, 7, figsize=(25, 25))
# drop the last two elements in ATTRS (cd and h)
ATTRS2 = ATTRS[:-2]
df_coniferous = df2[df2['class'].isin(coniferous)]
df_deciduous = df2[df2['class'].isin(deciduous)]
for i, attr1 in enumerate(ATTRS2):
    for j, attr2 in enumerate(ATTRS2):
        ax = axs[i, j]
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_coniferous, ax=ax, s=8, alpha=0.5, color='red')
        sns.scatterplot(x=f'latent_{attr1}', y=f'latent_{attr2}',
                        data=df_deciduous, ax=ax, s=8, alpha=0.5, color='blue')
        fontsize = 22
        # set the distance between the title and the plot
        # ax.set_title(f'{attr1} v. {attr2}', fontsize=fontsize, pad=10)
        ax.set_xlabel(attr1, fontsize=fontsize)
        ax.set_ylabel(attr2, fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'scatter_realset_vars_pairs_corr_coniferous_v_deciduous.png'))

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
"""
# Scatter plot of the input and reconstructed bands
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
for i, band in enumerate(S2_BANDS):
    ax = axs[i//4, i % 4]
    # adjust the point size and alpha and color
    sns.scatterplot(x='target_'+band, y='output_'+band,
                    data=df1, ax=ax, s=8, alpha=0.5)
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
plt.savefig(os.path.join(SAVE_PATH, 'linescatter_realset_bands_wocorr_target_v_output.png'))
plt.show()

# %%
"""
For each variable, cluster the sample according to forest type and then
according to the date of acquisition. Get the mean and std of the clustered
samples and then plot the time series of the mean and show the std as error bars.
"""
fig, axs = plt.subplots(3, 3, figsize=(22, 15))
for i, attr in enumerate(ATTRS):
    ax = axs[i//3, i % 3]
    # get the time seris of mean and std of the clustered samples for each variable
    mean_coniferous = []
    std_coniferous = []
    mean_deciduous = []
    std_deciduous = []
    for date in dates:
        df_filtered = df1[df1['date']==date]
        mean_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].mean())
        std_coniferous.append(df_filtered[df_filtered['class'].isin(coniferous)][f'latent_{attr}'].std())
        mean_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].mean())
        std_deciduous.append(df_filtered[df_filtered['class'].isin(deciduous)][f'latent_{attr}'].std())
    
    # plot the time series of the mean and show the std as error bars
    # Plot only the month and day of the date (original format 'yyyy.mm.dd', to this formar like dd/mm)
    dates_plot = [date.split('.')[2]+'/'+date.split('.')[1] for date in dates]
    ax.errorbar(x=dates_plot, y=mean_coniferous, yerr=std_coniferous, fmt='o', color = 'red', label='Coniferous')
    ax.errorbar(x=dates_plot, y=mean_deciduous, yerr=std_deciduous, fmt='o', color = 'blue', label='Deciduous')
    fontsize = 18
    ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel(attr, fontsize=fontsize)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_vars_wocorr_coniferous_v_deciduous.png'))
plt.show()

# %% NEW randomly select five samples and plot the time series of the mean and show the std as error bars for AE_RTM_corr
# randomly select five samples
sample_ids = np.random.choice(df2['sample_id'].unique(), 10)
for sample_id in sample_ids:
    fig, axs = plt.subplots(3, 3, figsize=(22, 15))
    for i, attr in enumerate(ATTRS):
        ax = axs[i//3, i % 3]
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
    sample_forest = 'Coniferous' if df2[df2['sample_id']==sample_id]['class'].values[0] in coniferous else 'Deciduous'
    sample_species = df2[df2['sample_id']==sample_id]['class'].values[0]
    # Set the title of the figure
    plt.suptitle(f"{sample_forest}: {sample_species}", fontsize=22)
    # Set the distance between the title and the plot
    plt.subplots_adjust(top=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'timeseries_realset_vars_corr_{sample_forest}_{sample_species}_{sample_id}.png'))

# %% NEW plot the time series of the mean and show the std as error bars for AE_RTM_corr
fig, axs = plt.subplots(3, 4, figsize=(28, 15))
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
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'timeseries_realset_bias_corr_coniferous_v_deciduous.png'))

# %%
