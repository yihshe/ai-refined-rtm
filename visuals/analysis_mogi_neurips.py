# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pykalman import KalmanFilter
# update plt to allow latex bold font
# plt.rcParams['text.usetex'] = True
# %%
MEAN = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_mean.npy')
SCALE = np.load('/maps/ys611/ai-refined-rtm/data/mogi/train_x_scale.npy')
mogi_paras = json.load(open(
    '/maps/ys611/ai-refined-rtm/configs/mogi/mogi_paras.json'))
# update dV to 10^6 m^3 scale for better visualization
mogi_paras['dV'] = {'min': -10, 'max': 10}
# ATTRS = list(mogi_paras.keys())
ATTRS = {
    'xcen': '$Z_{\mathrm{x_m}}$ (km)', 
    'ycen': '$Z_{\mathrm{y_m}}$ (km)', 
    'd': '$Z_{\mathrm{d}}$ (km)', 
    'dV': '$Z_{\mathrm{\Delta V}}$ ($10^6~\mathrm{m^3}$)'
}
assert list(ATTRS.keys()) == list(mogi_paras.keys())
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
def recale_output(df, mean, scale, corr=True):
    for attr in ['xcen', 'ycen', 'd']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / 1000 # scale back to km
    for attr in ['dV']:
        df[f'latent_{attr}'] = df[f'latent_{attr}'] / np.power(10, 6) # scale back to 10^6 m^3
    for i, gps in enumerate(GPS):
        df[f'output_{gps}'] = df[f'output_{gps}'] * scale[i] + mean[i]
        df[f'target_{gps}'] = df[f'target_{gps}'] * scale[i] + mean[i]
        if corr:
            df[f'init_output_{gps}'] = df[f'init_output_{gps}'] * scale[i] + mean[i]
            df[f'bias_{gps}'] = df[f'bias_{gps}'] * scale[i]
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
    return df

def r_square(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# define an exponential moving average function to fit the 1D time series data
def ema_filter(data, alpha=0.2):
    ema = [data[0]]  # Initialize EMA with first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

# define a kalman filter to fit the 1D time series

def kalman_filter(data, alpha=0.002):
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=alpha)
    state_means, _ = kf.filter(data)
    smooth_state_means, _ = kf.smooth(data)
    return smooth_state_means

# %%
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/'
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi/0509_103601_smooth'
# TODO add comparison results for wosmooth case in appendix
# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0509_103248_wosmooth'

CSV_PATH0 = os.path.join(
    BASE_PATH, 'AE_Mogi_corr/0509_102619_smooth', 
    'model_best_testset_analyzer_test.csv'
)
CSV_PATH1 = os.path.join(
    BASE_PATH, 'AE_Mogi/0509_103601_smooth',
    'model_best_testset_analyzer_test.csv'
)

# BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/mogi/models/AE_Mogi_corr/0509_103248_wosmooth'
CSV_PATH3 = os.path.join(
    BASE_PATH, 'AE_Mogi_corr/0509_103248_wosmooth',
    'model_best_testset_analyzer_test.csv'
)

df0 = recale_output(pd.read_csv(CSV_PATH0), MEAN, SCALE)
df1 = recale_output(pd.read_csv(CSV_PATH1), MEAN, SCALE, corr=False) 
df2 = recale_output(pd.read_csv(CSV_PATH3), MEAN, SCALE)

SAVE_PATH = os.path.join(BASE_PATH, 'AE_Mogi_corr/0509_102619_smooth', 
                         'neurips521')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

INFLATION_START = pd.Timestamp('2008-03-01')
INFLATION_END = pd.Timestamp('2008-07-01')

# %% Plot the line scatter for the output and target of each GPS
# df = df0
# for direction in ['ux', 'uy', 'uz']:
for direction in ['uz']:
    # fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    # for i, station in enumerate(station_info.keys()):
        # ax = axs[i//4, i % 4]
    i = 0
    for station in ['AV08', 'AV12']:
        for df in [df0, df1]:
            if i % 2 == 0:
                color = 'blue'
                label = 'w/ $\mathbf{C}$'
            else:
                color = 'red'
                label = 'w/o $\mathbf{C}$'
            ax = axs[i]
            i += 1
            gps = f'{direction}_{station}'
            sns.scatterplot(x='target_'+gps, y='output_'+gps, data=df, ax=ax, s=8,
                            alpha=0.5, color=color, label=label)
            # rmse = np.sqrt(np.mean((df[f'target_{gps}'] - df[f'output_{gps}'])**2))
            # NOTE calculate the R-square to measure the linearity of the data
            r2 = r_square(df[f'target_{gps}'], df[f'output_{gps}'])
            fontsize = 30
            ax.set_title(station, fontsize=fontsize)
            xlabel = '$X_{\mathrm{GPS}}$'
            ylabel = '$X_{\mathrm{GPS, B}}$' if i%2 == 0 else '$X_{\mathrm{GPS, C}}$'
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
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
            # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
            ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_PATH, f'{direction}_output_target_corr_v_wocorr.png'))
    plt.show()

# %% Plot the line scatter for the output and target of each GPS
# NOTE neurips full plot
df = df1
color = 'red'
ylabel = '$X_{\mathrm{GPS, C}}$'
for direction in ['ux', 'uy', 'uz']:
    fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//4, i % 4]
        gps = f'{direction}_{station}'
        sns.scatterplot(x='target_'+gps, y='output_'+gps, data=df, ax=ax, s=8,
                        alpha=0.5, color=color)
        # rmse = np.sqrt(np.mean((df[f'target_{gps}'] - df[f'output_{gps}'])**2))
        r2 = r_square(df[f'target_{gps}'], df[f'output_{gps}'])
        fontsize = 30
        ax.set_title(station, fontsize=fontsize)
        xlabel = '$X_{\mathrm{GPS}}$'
        # ylabel = '$X_{\mathrm{GPS, B}}$'
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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
        # ax.legend([f'RMSE: {rmse:.3f}'], fontsize=24)
        ax.legend([f'$R^2$: {r2:.3f}'], fontsize=24)
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_PATH, f'{direction}_output_target_wocorr.png'))
    plt.show()

# %% plot the histogram of the four variables
# NOTE plot with w/o correction case
NUM_BINS = 100
# df = df0
fig, axs = plt.subplots(1, 4, figsize=(25, 5))
for i, attr in enumerate(ATTRS.keys()):
    ax = axs[i]
    sns.histplot(
        df1[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='red',
        alpha=0.5,
        # label='w/o $C$'
        # set bold font for the latex label $C$
        label='w/o $\mathbf{C}$'
    )
    sns.histplot(
        df0[f'latent_{attr}'].values,
        bins=NUM_BINS,
        ax=ax,
        color='blue',
        alpha=0.5,
        label='w/ $\mathbf{C}$'
    )
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    fontsize = 30
    ax.set_xlabel(ATTRS[attr], fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.yaxis.labelpad = 10
    ax.legend(fontsize=fontsize-5)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'histogram_latent_corr_v_wocorr.png'))
plt.show()

# %%
# scatter plot of the latent variables given date order
# first sort the dataframe by date in ascending order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# df = df.sort_values(by='date')
fig, axs = plt.subplots(2, 2, figsize=(20, 8))
for i, attr in enumerate(ATTRS.keys()):
    ax = axs[i//2, i % 2]
    # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
    ax.axvspan(
        INFLATION_START, INFLATION_END,
        color='pink', alpha=0.4
    )
    sns.scatterplot(
        x='date', y=f'latent_{attr}', data=df, ax=ax, s=8, alpha=0.8, color='blue'
    )
    # plot the fitted curve for only dV
    smooth_state_means = kalman_filter(
        df[f'latent_{attr}'].values, alpha=0.002)
    ax.plot(df['date'], smooth_state_means, color='red', linewidth=2)

    fontsize = 32
    # ax.set_title(attr, fontsize=fontsize)
    ax.set_xlabel('Date', fontsize=fontsize)
    # show only years for Date
    ax.set_ylabel(ATTRS[attr], fontsize=fontsize)
    ax.set_ylim(mogi_paras[attr]['min'], mogi_paras[attr]['max'])
    ax.tick_params(axis='both', which='major', labelsize=25) #25
    # rotate the x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(-20)
    
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'latent_vars_date.png'))
plt.show()


# %%
# scatter plot of the gps displacement given date order for both output and target
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
# for direction in ['ux', 'uy', 'uz']:
for direction in ['uy']:
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
            tick.set_rotation(-20)
    plt.tight_layout()
    # plt.savefig(os.path.join(
    #     SAVE_PATH, f'{direction}_target_v_finaloutput_gps_date.png'))
    plt.show()

"""
The following code is for AE_Mogi_corr only
"""
# %%
# scater plot of the corrected bias given date order
df = df0
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
for direction in ['ux', 'uy', 'uz']:
# for direction in ['uy']:
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))
    for i, station in enumerate(station_info.keys()):
        ax = axs[i//3, i % 3]
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'bias_{gps}', data=df, ax=ax, s=10, alpha=0.5
        )
        fontsize = 32
        # smooth_bias = kalman_filter(df[f'bias_{gps}'].values, alpha=0.01)
        # ax.plot(df['date'], smooth_bias, color='red', linewidth=2)
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
    plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_PATH, f'{direction}_bias_gps_date.png'))
    plt.show()
# %%
# plot the scatter plot of the init_output from Mogi and target
df = df0
# fig, axs = plt.subplots(2, 3, figsize=(30, 10))
fig, axs = plt.subplots(12, 3, figsize=(30, 60))
for i, direction in enumerate(['ux', 'uy', 'uz']):
    # for j, station in enumerate(['AV08', 'AV12']):
    for j, station in enumerate(station_info.keys()):
        ax = axs[j, i]
        # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output
        smooth_init_output = kalman_filter(df[f'init_output_{gps}'].values, alpha=0.005)
        ax.plot(df['date'], smooth_init_output, color='red', linewidth=2)

        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # set y limit same so easier for visual
        y_min = -22 if direction == 'uz' else -12
        y_max = 22 if direction == 'uz' else 12
        ax.set_ylim(y_min, y_max)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, f'{direction}_target_v_mogioutput_gps_date_AV08_12.png'))
plt.savefig(os.path.join(
    SAVE_PATH, 'target_v_mogioutput_gps_date.png'))
plt.show()

# %% NOTE comparing the effects of the smoothness term
# plot the scatter plot of the init_output from Mogi and target
# df = df0
station = 'AV08'
fig, axs = plt.subplots(2, 3, figsize=(30, 10))
for i, direction in enumerate(['ux', 'uy', 'uz']):
    # for j, station in enumerate(['AV08', 'AV12']):
    for j, df in enumerate([df0, df2]):
        ax = axs[j, i]
        # plot the background of the plot as grey for date between 2008-02-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        gps = f'{direction}_{station}'
        sns.scatterplot(
            x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='orange'
        )
        sns.scatterplot(
            x='date', y=f'init_output_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output
        smooth_init_output = kalman_filter(df[f'init_output_{gps}'].values, alpha=0.005)
        ax.plot(df['date'], smooth_init_output, color='red', linewidth=2)

        fontsize = 32
        title  = f'{station} (w/o smoothness)' if j == 1 else f'{station} (w/ smoothness)'
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # set y limit same so easier for visual
        y_min = -22 if direction == 'uz' else -12
        y_max = 22 if direction == 'uz' else 12
        ax.set_ylim(y_min, y_max)
        # ax.legend(fontsize=fontsize)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, f'target_v_mogioutput_gps_date_AV08_smoothness_ablation.png'))
plt.show()
# %%
# combine csvs from both train and test sets to plot the gps displacements
CSV_PATH = os.path.join(
    BASE_PATH, 'AE_Mogi_corr/0509_102619_smooth', 
    'model_best_testset_analyzer_train.csv'
)
df0_train = recale_output(pd.read_csv(CSV_PATH), MEAN, SCALE)
df0_full = pd.concat([df0, df0_train], axis=0)
# select the data from df0_full up to 2014-01-01
df0_full = df0_full[df0_full['date'] < pd.Timestamp('2013-06-01')]
df= df0_full
# get the last date in the test set
last_date = df0['date'].max() # 2009-07-29
# for AV08, AV12 (each row), plot the gps displacements of 'uz' in each column
# column 1: corrected_output v. target, column 2: init_output v. target, column 3: bias
# fig, axs = plt.subplots(2, 3, figsize=(30, 10))
fig, axs = plt.subplots(12, 3, figsize=(30, 60))
direction = 'ux'
# for i, station in enumerate(['AV08', 'AV12']):
for i, station in enumerate(station_info.keys()):
    for j, attr in enumerate(['init_output', 'bias', 'output']):
        ax = axs[i, j]
        gps = f'{direction}_{station}'
        # plot the background of the plot as pink for date between 2008-03-01 to 2008-07-01
        ax.axvspan(
            INFLATION_START, INFLATION_END,
            color='pink', alpha=0.4
        )
        if attr in ['output', 'init_output']:
            sns.scatterplot(
                x='date', y=f'target_{gps}', data=df, ax=ax, s=15, alpha=0.5,
                color='orange'
            )
        sns.scatterplot(
            x='date', y=f'{attr}_{gps}', data=df, ax=ax, s=15, alpha=0.5,
            color='blue'
        )
        # kalman filter to fit the curve for init_output
        smooth_init_output = kalman_filter(df[f'{attr}_{gps}'].values, alpha=0.005)
        ax.plot(df['date'], smooth_init_output, color='red', linewidth=2)
        # plot a blue line to indicate the last date in the test set
        ax.axvline(x=last_date, color='grey', linestyle='-', linewidth=2)
        fontsize = 32
        ax.set_title(station, fontsize=fontsize)
        ax.set_xlabel('Date', fontsize=fontsize)
        ax.set_ylabel(dirs[direction], fontsize=fontsize)
        y_min = -21 if direction == 'uz' else -11
        y_max = 21 if direction == 'uz' else 11
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', which='major', labelsize=25)
        # rotate the x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(-20)
plt.tight_layout()
# plt.savefig(os.path.join(
#     SAVE_PATH, f'{direction}_mogi_v_bias_v_corr_gps_date_AV08_12.png'))
plt.savefig(os.path.join(
    SAVE_PATH, f'{direction}_mogi_v_bias_v_corr_gps_date.png'))
plt.show()
# %%
