import os
import pandas as pd
from sklearn.model_selection import train_test_split

# BASE_DIR = '/maps/ys611/ai-refined-rtm/data/'
# DATA_DIR = os.path.join(BASE_DIR, 'data/BPWW_extract_2018_reshaped.csv')
# SAVE_DIR_TRAIN = os.path.join(BASE_DIR,
#                               'data/BPWW_extract_2018_reshaped_train.csv')
# SAVE_DIR_TEST = os.path.join(BASE_DIR,
#                              'data/BPWW_extract_2018_reshaped_test.csv')

# suffix = "all_CA_range_norm"
# BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20230715/'
# DATA_DIR = os.path.join(BASE_DIR, f'synthetic_{suffix}.csv')
# SAVE_DIR = os.path.join(BASE_DIR, suffix)

BASE_DIR = '/maps/ys611/ai-refined-rtm/data/synthetic/20230816/'
DATA_DIR = os.path.join(BASE_DIR, 'synthetic_rand_noise.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'synthetic_rand_noise')
SAVE_DIR_TRAIN = os.path.join(SAVE_DIR, 'synthetic_train_valid.csv')
SAVE_DIR_TEST = os.path.join(SAVE_DIR, 'synthetic_test.csv')

SPLIT_RATIO = 0.2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def split():
    # read the data
    data = pd.read_csv(DATA_DIR)
    # split the data into train and test sets, set the split ratio to 0.2
    train, test = train_test_split(
        data, test_size=SPLIT_RATIO, random_state=42)

    # save the train and test sets
    train.to_csv(SAVE_DIR_TRAIN, index=False)
    test.to_csv(SAVE_DIR_TEST, index=False)


if __name__ == '__main__':
    split()
