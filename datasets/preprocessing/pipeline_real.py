"""
Create a pipeline for preprocessing the real dataset. 
Step 1: split the data into train, validation, and test sets
Step 2: reshape the data by breaking the time series into multiple samples
Step 3: standardize the data
"""
#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%%
# fix random seeds for reproducibility
SEED = 123
np.random.seed(SEED)
BASE_DIR = '/maps/ys611/ai-refined-rtm/data/real/'
DATA_DIR = os.path.join(BASE_DIR, 'BPWW_extract_2018.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'split_embed')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']

# get the unique tree species
Species = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra', 
           'Larix decidua', 'Pinus sylvestris', 'Prunus spp', 
           'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp', 
           'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']
Coniferous = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra', 
              'Larix decidua', 'Pinus sylvestris']
Deciduous = ['Prunus spp', 'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp', 
             'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']
Dates = ['2018.04.08', '2018.04.21', '2018.05.06', '2018.07.02', '2018.08.09', 
         '2018.08.21', '2018.08.29', '2018.09.13', '2018.09.18', '2018.09.28', 
         '2018.09.30', '2018.10.05', '2018.10.10', '2018.10.30']

#%%
def encode_dates(df):
    # Convert to pandas datetime
    dates = pd.to_datetime(df['date'], format='%Y.%m.%d')
    # Get day of the year
    df['day_of_year'] = dates.dt.dayofyear
    max_day_of_year = 366
    # Cyclical transformation
    df['sin_date'] = np.sin(2 * np.pi * df['day_of_year'] / max_day_of_year)
    df['cos_date'] = np.cos(2 * np.pi * df['day_of_year'] / max_day_of_year)
    df = df.drop(columns='day_of_year')
    return df

def encode_species(df):
    # encode species
    species = df['class'].astype('category')
    df['species_idx'] = species.cat.codes
    # encode forest group
    df['group_idx'] = df['class'].isin(Coniferous).astype(int)
    return df

#%% reshape the data
def reshape(df):
    # extract dates information from df.columns
    dates = [col.split('_')[0]
            for col in df.columns if len(col.split('_')) == 3]
    dates = list(set(dates))
    dates.sort()

    # reshape the data
    data = None
    for date in dates:
        fields = [date+'_'+field for field in S2_BANDS]
        # scale the reflectance by 10000
        df_temp = df[fields]/10000.0
        df_temp['class'] = df['class']
        df_temp['sample_id'] = df['sample_id']
        df_temp['date'] = date
        df_temp = encode_dates(df_temp)
        df_temp = encode_species(df_temp)
        # concatenate df_temp.values to data
        if data is None:
            data = df_temp.values
        else:
            data = np.concatenate((data, df_temp.values), axis=0)
    # shuffle the rows of data
    np.random.shuffle(data)
    # create a new dataframe with unique bands as column
    return pd.DataFrame(data, columns=S2_BANDS+
                        ['class', 'sample_id', 'date', 
                         'sin_date', 'cos_date', 'species_idx', 'group_idx'])

#%% load the real data
df = pd.read_csv(DATA_DIR, delimiter=';')
# assign unique id to each row
if 'sample_id' not in df.columns:
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample_id'}, inplace=True)

#%% split the data into train, validation, and test sets
SPLIT_RATIO = 0.2
train, test = train_test_split(
    df, test_size=SPLIT_RATIO, random_state=42)
train, valid = train_test_split(
    train, test_size=SPLIT_RATIO, random_state=42)

#%% reshape the data by breaking the time series into multiple samples
train = reshape(train)
valid = reshape(valid)
test = reshape(test)

#%% standardize the data
scaler = preprocessing.StandardScaler().fit(train[S2_BANDS])
train[S2_BANDS] = scaler.transform(train[S2_BANDS])
valid[S2_BANDS] = scaler.transform(valid[S2_BANDS])
test[S2_BANDS] = scaler.transform(test[S2_BANDS])

#%% save the scaler and the data
np.save(os.path.join(SAVE_DIR, 'train_x_mean.npy'), scaler.mean_)
np.save(os.path.join(SAVE_DIR, 'train_x_scale.npy'), scaler.scale_)
train.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
valid.to_csv(os.path.join(SAVE_DIR, 'valid.csv'), index=False)
test.to_csv(os.path.join(SAVE_DIR, 'test.csv'), index=False)
#%%
"""
Rescale the real test set using the scaler from the synthetic dataset
This test set will be used to evaluate the model trained on the synthetic dataset
"""
# test = pd.read_csv(os.path.join(SAVE_DIR, 'test.csv'))
# scaler = {
#     'real':{
#         'mean': np.load(os.path.join('/maps/ys611/ai-refined-rtm/data/real', 
#                                      'train_x_mean.npy')),
#         'scale': np.load(os.path.join('/maps/ys611/ai-refined-rtm/data/real',
#                                       'train_x_scale.npy'))
#     },
#     'synthetic':{
#         'mean': np.load(
#             os.path.join('/maps/ys611/ai-refined-rtm/data/synthetic/20240124', 
#                          'train_x_mean.npy')),
#         'scale': np.load(
#             os.path.join('/maps/ys611/ai-refined-rtm/data/synthetic/20240124', 
#                          'train_x_scale.npy'))
#     }
# }
# # rescale the real test set
# test[S2_BANDS] = (test[S2_BANDS] * scaler['real']['scale']) + scaler['real']['mean']
# test[S2_BANDS] = (test[S2_BANDS] - scaler['synthetic']['mean']) / scaler['synthetic']['scale']
# test.to_csv(os.path.join(SAVE_DIR, 'test_syn.csv'), index=False)

# %%
