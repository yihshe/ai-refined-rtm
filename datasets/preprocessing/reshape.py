"""
Run this script to reshape the BPWW_extract_2018.csv file.
The reshaped file will be used for training the AutoEncoder.
"""
import numpy as np
import pandas as pd
import pdb
pdb.set_trace()

CSV_PATH = '/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018.csv'
SAVE_PATH = '/maps/ys611/ai-refined-rtm/data/real/BPWW_extract_2018_reshaped.csv'


def preprocess():
    df = pd.read_csv(CSV_PATH, delimiter=';')
    # assign unique id to each row
    if 'sample_id' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'sample_id'}, inplace=True)

    # extract dates information from df.columns
    dates = [col.split('_')[0]
             for col in df.columns if len(col.split('_')) == 3]
    # extract the unique dates
    unique_dates = list(set(dates))
    # extract the unique dates and sort them
    unique_dates.sort()

    # extract bands information from df.columns
    bands = [col.split('_')[1]+'_'+col.split('_')[2]
             for col in df.columns if len(col.split('_')) == 3]
    # extract the unique bands
    unique_bands = list(set(bands))
    # extract the unique bands and sort them
    unique_bands.sort()

    unique_vars = ['FAPAR', 'LAI']

    # create an empty numpy array to store the data
    data = None
    for date in unique_dates:
        fields = [date+'_'+field for field in unique_bands+unique_vars]
        fields += ['class', 'sample_id']
        # query df with the fields and add a column with the date
        df_temp = df[fields]
        df_temp['date'] = date
        # concatenate df_temp.values to data
        if data is None:
            data = df_temp.values
        else:
            data = np.concatenate((data, df_temp.values), axis=0)

    # create a new dataframe with unique bands as column
    df_new = pd.DataFrame(data, columns=unique_bands+unique_vars +
                          ['class', 'sample_id', 'date'])

    # save the reshaped dataframe
    # df_new.to_csv(SAVE_PATH, index=False)


if __name__ == '__main__':
    preprocess()
