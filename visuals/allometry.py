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
    plt.xlabel('Crown Diameter (CD)')
    plt.ylabel('Height (H)')
    plt.title(f'{field} {zone}: H = exp({np.round(params[0], 3)}+{np.round(params[1], 3)}*ln(CD))')
    plt.savefig(SAVE_PATH + field +'_'+ zone + '_H_vs_CD_fit.png')
    plt.show()


# %%
# Plot for all biomes
# field = 'Biogeographic_zone'
field = 'Biome'
zones_of_interest = df[field].unique()
zones = ['Temperate mixed forests', 'Tropical forests',
       'Temperate coniferous forests', 'Boreal forests', 'Woodlands and savannas']

# Filter the data by biogeographic zones
filtered_df = df[df[field].isin(zones)]
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
plt.xlabel('Crown Diameter (CD)')
plt.ylabel('Height (H)')
plt.title(f'{field} All incl Woodlands and savannas: H = exp({np.round(params[0], 3)}+{np.round(params[1], 3)}*ln(CD))')
plt.savefig(SAVE_PATH + field +'_'+ 'All_incl_woodlands' + '_H_vs_CD_fit.png')
plt.show()
# %%
