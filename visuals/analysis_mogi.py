# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
# %%
MEAN = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_mean.npy')
SCALE = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_scale.npy')
mogi_paras = json.load(open(
    '/maps/ys611/ai-refined-rtm/configs/mogi/mogi_paras.json'))
station_info = json.load(open(
    '/maps/ys611/ai-refined-rtm/configs/mogi/station_info.json'))
GPS = []
for direction in ['ux', 'uy', 'uz']:
    for station in station_info.keys():
        GPS.append(f'{direction}_{station}')
# %%
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0503_093049_reg1e-2'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0503_093622_reg1e-1'
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0505_161157_std_loss'

k = 'train'
CSV_PATH0 = os.path.join(
    BASE_PATH, f'model_best_testset_analyzer_{k}.csv'
)
df0 = pd.read_csv(CSV_PATH0)

SAVE_PATH = os.path.join(BASE_PATH, f'plots_{k}')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# %% Plot the line scatter for the output and target of each GPS
df = df0
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(x='target_'+gps, y='output_'+gps, data=df, ax=ax, s=8,
                        alpha=0.5)
        fontsize = 16
        ax.set_title(gps, fontsize=fontsize)
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
    plt.savefig(os.path.join(SAVE_PATH, f'{direction}_output_target.png'))
    plt.show()


# %% plot the histogram of the six variables
NUM_BINS = 100
# ATTRS = ['xcen', 'ycen', 'd', 'dV_factor', 'dV_power', 'dV']
ATTRS = ['xcen', 'ycen', 'd', 'dV']
# ATTRS = ['1', '2', '3', '4', '5']
df = df0
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, attr in enumerate(ATTRS):
    ax = axs[i//3, i % 3]
    sns.histplot(
        df[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        alpha=0.5,
    )
    ax.tick_params(axis='both', which='major', labelsize=16)
    fontsize = 18
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'histogram_latent.png'))

# %%
# scatter plot of the latent variables given date order
# first sort the dataframe by date in ascending order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# df = df.sort_values(by='date')
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, attr in enumerate(ATTRS):
    ax = axs[i//3, i % 3]
    sns.scatterplot(
        x='date', y=f'latent_{attr}', data=df, ax=ax, s=8, alpha=0.5
    )
    fontsize = 16
    ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel('Latent', fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'latent_vars_date.png'))
plt.show()

# # %%
# # scatter plot of the gps displacement given date order
# # first sort the dataframe by date in ascending order
# df = df0
# # convert date from string to decimal
# df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# # df = df.sort_values(by='date')
# for direction in ['ux', 'uy', 'uz']:
#     fig, axs = plt.subplots(3, 4, figsize=(25, 15))
#     for i, station in enumerate(station_info.keys()):
#         ax = axs[i//4, i % 4]
#         gps = f'{direction}_{station}'
#         sns.scatterplot(
#             x='date', y=f'output_{gps}', data=df, ax=ax, s=8, alpha=0.5
#         )
#         fontsize = 16
#         ax.set_title(gps, fontsize=fontsize)
#         ax.set_xlabel('Date', fontsize=fontsize)
#         ax.set_ylabel('Displacement', fontsize=fontsize)
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_PATH, f'{direction}_output_gps_date.png'))
#     plt.show()


# %%
# scatter plot of the gps displacement given date order for both output and target
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(25, 15))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'output_{gps}', data=df, ax=ax, s=10, alpha=0.5, label='output'
        )
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=10, alpha=0.5, label='target'
        )
        fontsize = 16
        ax.set_title(gps, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel('Displacement', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'{direction}_output_target_gps_date.png'))
    plt.show()

"""
The following code is for AE_Mogi_corr only
"""
# %%
# scater plot of the corrected bias given date order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(25, 15))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'bias_{gps}', data=df, ax=ax, s=10, alpha=0.5
        )
        fontsize = 16
        ax.set_title(gps, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel('Bias', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'{direction}_bias_gps_date.png'))
    plt.show()
# %%
# plot the scatter plot of the init_output from Mogi and target
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(25, 15))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=10, alpha=0.5,
            label='init_output'
        )
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=10, alpha=0.5,
            label='target'
        )
        fontsize = 16
        ax.set_title(gps, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel('Displacement', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'{direction}_init_output_v_target_gps_date.png'))
    plt.show()
# %%
