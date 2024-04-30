#%%
import h5py
import json
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
BASE_PATH = '/maps/ys611/ai-refined-rtm/'
file = h5py.File(path.join(BASE_PATH, 'data/mogi/ts_filled_ICA9comp.mat'))
#%%
# load the time series data, 6525*36, 6525 time steps, 12*3 observations at each step
ts = file['Xd_filled']['ts'][()]
timeline = file['Xd_filled']['timeline'][()]

#%%
# load the name for each displacement
name_refs = file['Xd_filled']['name'][()][0]
name_strs = []
for i in range(len(name_refs)):
    name_int = file[name_refs[i]][:].flatten()
    name_str = ''.join([chr(i) for i in name_int if 0 <= i < 256])
    name_strs.append(name_str)
# get the 12 station names and 3 directions
station_names = [i[:4] for i in name_strs[::3]]
directions = [i[-1] for i in name_strs[:3]] # ['e', 'n', 'u']
#%%
# get the locations of the stations
station_dict = {}
for i, (xE, yN, zV) in enumerate(zip(
    file['stations']['xE'][()].flatten(),
    file['stations']['yN'][()].flatten(),
    file['stations']['zV'][()].flatten()
)):
    station_dict[station_names[i]] = {
        'xE': xE,
        'yN': yN,
        'zV': zV
    }
# save station_dict to a json file
with open(path.join(BASE_PATH, 'data/mogi/station_info.json'), 'w') as f:
    json.dump(station_dict, f, indent=2)

#%%
# save the data to a csv file 
for i, displacement in enumerate(['ux', 'uy', 'uz']):
    # get the data from ts and create column names
    data = ts[:, i::3]
    columns = [f'{displacement}_{j}' for j in station_names]
    if i == 0:
        df = pd.DataFrame(data, columns=columns)
    else:
        df[columns] = pd.DataFrame(data)
#%%
# add the timeline to the dataframe
df['date'] = timeline
# save the dataframe to a csv file
df.to_csv(path.join(BASE_PATH, 'data/mogi/ts_filled_ICA9comp.csv'), index=False)

#%%
# Plot the time series data for each displacement of the first station
for station_name in station_names:
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    # station_name = station_names[0]
    for i, displacement in enumerate(['ux', 'uy', 'uz']):
        ax[i].plot(df['date'], df[f'{displacement}_{station_name}'])
        ax[i].set_title(f'{displacement} for station {station_name}')
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel('Displacement (m)')

#%%
# Plot the histogram for each displacement of the first station
for station_name in station_names:
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    # station_name = station_names[0]
    for i, displacement in enumerate(['ux', 'uy', 'uz']):
        ax[i].hist(df[f'{displacement}_{station_name}'], bins=100)
        ax[i].set_title(f'{displacement} for station {station_name}')
        ax[i].set_xlabel('Displacement (m)')
        ax[i].set_ylabel('Frequency')

#%% 
