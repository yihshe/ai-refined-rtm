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
# update dV to 10^6 m^3 scale for better visualization
mogi_paras['dV'] = {'min': -10, 'max': 10}
ATTRS = list(mogi_paras.keys())
station_info = json.load(open(
    '/maps/ys611/ai-refined-rtm/configs/mogi/station_info.json'))
STATIONS = list(station_info.keys())
DIRECTIONS = ['ux', 'uy', 'uz']
dirs = {
    'ux': 'East (mm)',
    'uy': 'North (mm)',
    'uz': 'Vertical (mm)'
}
GPS = []
for direction in DIRECTIONS:
    for station in STATIONS:
        GPS.append(f'{direction}_{station}')
# rescale the output to the original scale
def recale_output(df, mean, scale):
    for attr in ['xcen', 'ycen', 'd']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / 1000 # scale back to km
    for attr in ['dV']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / np.power(10, 6) # scale back to 10^6 m^3
    for i, gps in enumerate(GPS):
        df[f'init_output_{gps}'] = df[f'init_output_{gps}'] * scale[i] + mean[i]
        df[f'output_{gps}'] = df[f'output_{gps}'] * scale[i] + mean[i]
        df[f'target_{gps}'] = df[f'target_{gps}'] * scale[i] + mean[i]
        df[f'bias_{gps}'] = df[f'bias_{gps}'] * scale[i]
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
    return df

# %%
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0506_163645'

k = 'valid'
CSV_PATH0 = os.path.join(
    BASE_PATH, f'model_best_testset_analyzer_{k}.csv'
)

df0 = recale_output(pd.read_csv(CSV_PATH0), MEAN, SCALE)

SAVE_PATH = os.path.join(BASE_PATH, f'plots_{k}')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# %% Plot the line scatter for the output and target of each GPS
df = df0
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(x='target_'+gps, y='output_'+gps, data=df, ax=ax, s=8,
                        alpha=0.5)
        rmse = np.sqrt(np.mean((df[f'target_{gps}'] - df[f'output_{gps}'])**2))
        fontsize = 30
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Input', fontsize=fontsize)
        ax.set_ylabel('Reconstruction', fontsize=fontsize)
        # set the same ticks for both x and y axes
        ax.tick_params(axis='both', which='major', labelsize=25)
        # plot the diagonal line
        limits = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
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
        ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'{direction}_output_target.png'))
    plt.show()


# %% plot the histogram of the six variables
NUM_BINS = 100
# ATTRS = ['1', '2', '3', '4', '5']
df = df0
fig, axs = plt.subplots(1, 4, figsize=(25, 5))
for i, attr in enumerate(ATTRS):
    ax = axs[i]
    sns.histplot(
        df[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        alpha=0.5,
    )
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    fontsize = 30
    ax.set_xlabel(attr, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.yaxis.labelpad = 10
    # ax.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'histogram_latent.png'))
plt.show()
# %%
# scatter plot of the latent variables given date order
# first sort the dataframe by date in ascending order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# df = df.sort_values(by='date')
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
for i, attr in enumerate(ATTRS):
    ax = axs[i//2, i % 2]
    sns.scatterplot(
        x='date', y=f'latent_{attr}', data=df, ax=ax, s=8, alpha=0.5
    )
    fontsize = 32
    ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('Date', fontsize=fontsize)
    ax.set_ylabel(attr, fontsize=fontsize)
    ax.set_ylim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    ax.tick_params(axis='both', which='major', labelsize=25) #25
    # rotate the x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(-30)
    
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
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=10, alpha=0.5, 
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'output_{gps}', data=df, ax=ax, s=10, alpha=0.4, 
            color='blue'
        )
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-30)
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'{direction}_target_v_finaloutput_gps_date.png'))
    plt.show()

"""
The following code is for AE_Mogi_corr only
"""
# %%
# scater plot of the corrected bias given date order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'bias_{gps}', data=df, ax=ax, s=10, alpha=0.5
        )
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-30)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'{direction}_bias_gps_date.png'))
    plt.show()
# %%
# plot the scatter plot of the init_output from Mogi and target
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=15, alpha=0.3,
            color='blue'
        )
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-30)
    plt.tight_layout()
    # plt.savefig(os.path.join(
    #     SAVE_PATH, f'{direction}_target_v_mogioutput_gps_date.png'))
    plt.show()
# %%
