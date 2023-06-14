# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# %%
# plot the histogram of the training data
# CSV_PATH = '/maps/ys611/ai-refined-rtm/data/BPWW_extract_2018_reshaped_train_scaled.csv'
# SAVE_PATH = '/maps/ys611/ai-refined-rtm/visuals/histograms'

# # plot the histogram of reconstrucion error
# CSV_PATH = '/maps/ys611/ai-refined-rtm/saved/models/VanillaAE_scaled/0510_211403/model_best_loss_analyzer.csv'
# SAVE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/VanillaAE_scaled/0510_211403/histograms'

# plot the histogram of latent space
BASE_PATH = '/maps/ys611/ai-refined-rtm/saved/models/AE_RTM/0612_175828_/'
CSV_PATH = os.path.join(BASE_PATH, 'model_best_testset_analyzer.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'histograms')

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']
ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'sd', 'h', 'cd']

NUM_BINS = 100
# mkdir if the save path does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read the csv file
df = pd.read_csv(CSV_PATH)

# %%
# create individual figures for each band using seaborn
for band in S2_BANDS:
    sns.histplot(df[band].values, bins=NUM_BINS)
    # plt.title(f'Training Data: Histogram of {band}')
    # plt.xlabel('Reflectance')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.savefig(os.path.join(
    #     SAVE_PATH, f'histogram_trainset_scaled_{band}.png'))
    # plt.show()
    plt.title(f'Test Data: Histogram of {band}')
    plt.xlabel('Squared Reconstruction Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(
        SAVE_PATH, f'histogram_testset_scaled_loss_{band}.png'))
    plt.show()

# %%
# create one figure and plot each band as a subplot
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
for i, band in enumerate(S2_BANDS):
    #     sns.histplot(df[band].values, bins=NUM_BINS, ax=axs[i//4, i % 4])
    #     axs[i//4, i % 4].set_title(f'Histogram of {band}')
    #     axs[i//4, i % 4].set_xlabel('Reflectance')
    #     axs[i//4, i % 4].set_ylabel('Frequency')
    # plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_PATH, 'histogram_trainset_scaled.png'), dpi=300)
    # plt.show()
    sns.histplot(df[band].values, bins=NUM_BINS, ax=axs[i//4, i % 4])
    axs[i//4, i % 4].set_title(f'Histogram of {band}')
    axs[i//4, i % 4].set_xlabel('Squared Reconstruction Error')
    axs[i//4, i % 4].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_testset_scaled_loss.png'), dpi=300)
plt.show()

# %%
# create one figure and plot each attribute as a subplot
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
for i, attr in enumerate(ATTRS):
    sns.histplot(df[f'latent_{attr}'].values,
                 bins=NUM_BINS, ax=axs[i//3, i % 3])
    # axs[i//3, i % 3].set_title(f'{attr}')
    axs[i//3, i % 3].set_xlabel(attr)
    axs[i//3, i % 3].set_ylabel('Frequency')
    # adjust the font size of the x and y labels
    axs[i//3, i % 3].xaxis.label.set_size(18)
    axs[i//3, i % 3].yaxis.label.set_size(18)
plt.tight_layout()
plt.savefig(os.path.join(
    SAVE_PATH, 'histogram_testset_rtm_vars.png'), dpi=300)
plt.show()

# %%
