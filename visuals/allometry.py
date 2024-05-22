#%%
# Analyze the allometric database and plot the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np  
from scipy.optimize import curve_fit
#%%
# Define the paths to read the data and save the plots
DATA_PATH = '/maps/ys611/ai-refined-rtm/data/allometry/Data.csv'
SAVE_PATH = '/maps/ys611/ai-refined-rtm/visuals/allometry/'
# Create the save path if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
#%%
# TODO test with different biomes and zones
# Load the data
df = pd.read_csv(DATA_PATH)  # Replace with your CSV file path
# %%
# Define a power-law function to fit
def nonlin_func(x, a, b):
    return np.exp(a + b * np.log(x))
#%%
# Plot for each biome
field = 'Biogeographic_zone'
# field = 'Biome'
zones_of_interest = df[field].unique()
for zone in zones_of_interest:
    # Filter the data by biogeographic zones
    filtered_df = df[df[field].isin([zone])]
    # Fit your data to the log_func
    params, covariance = curve_fit(nonlin_func, filtered_df['CD'], filtered_df['H'])
    # Generate a sequence of crown diameters for plotting the fit line
    cd_space = np.linspace(min(filtered_df['CD']), max(filtered_df['CD']), 100)
    # Plot the original scatter data
    plt.figure(figsize=(10, 6))
    # plt.scatter(filtered_df['CD'], filtered_df['H'], alpha=0.5)
    plt.hexbin(x=filtered_df['CD'], y=filtered_df['H'], gridsize=80, cmap='viridis', bins='log')
    # Plot the logarithmic fit line
    plt.plot(cd_space, nonlin_func(cd_space, *params), color='red', linewidth=2)
    fontsize = 25
    plt.xlabel('Crown Diameter ($Z_{\mathrm{CD}}$)')
    plt.ylabel('Height ($Z_{\mathrm{H}}$)')
    plt.title(f'{field} {zone}: H = exp({np.round(params[0], 3)}+{np.round(params[1], 3)}*ln(CD))')
    plt.savefig(SAVE_PATH + field +'_'+ zone + '_H_vs_CD_fit.png')
    plt.show()


# %%
# Plot for all biomes
# field = 'Biogeographic_zone'
field = 'Biome'
zones_of_interest = df[field].unique()
zones = ['Temperate mixed forests', 'Temperate coniferous forests']
# Filter the data by biogeographic zones
filtered_df = df[df[field].isin(zones)]
X = 'CD'
Y = 'H'
labels = {
    'CD': 'Crown Diameter ($Z_{\mathrm{CD}}$)',
    'H': 'Tree Height ($Z_{\mathrm{H}}$)',
}

# Fit your data to the log_func
params, covariance = curve_fit(nonlin_func, filtered_df[X], filtered_df[Y])

# Calculate the R-squared value
residuals = filtered_df[Y] - nonlin_func(filtered_df[X], *params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((filtered_df[Y]-np.mean(filtered_df[Y]))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f'R-squared: {r_squared}')

# Generate a sequence of crown diameters for plotting the fit line
cd_space = np.linspace(min(filtered_df[X]), max(filtered_df[X]), 100)
# Plot the original scatter data
plt.figure(figsize=(10, 6))
# plt.scatter(filtered_df['CD'], filtered_df['H'], alpha=0.5)
plt.hexbin(x=filtered_df[X], y=filtered_df[Y], gridsize=80, cmap='viridis', bins='log')
# Plot the logarithmic fit line
plt.plot(cd_space, nonlin_func(cd_space, *params), color='red', linewidth=2)
fontsize = 22
plt.xlabel(labels[X], fontsize=fontsize)
plt.ylabel(labels[Y], fontsize=fontsize)
# set the tick labels size same for both axes
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.title(f'{field} Temperate Forests: {Y} = exp({np.round(params[0], 3)}+{np.round(params[1], 3)}*ln({X})), R-squared: {np.round(r_squared, 3)}')
plt.tight_layout()
plt.savefig(SAVE_PATH + field +'_'+ 'temperate_forests' + f'_{X}_vs_{Y}_fit.png')
plt.show()

# %%
