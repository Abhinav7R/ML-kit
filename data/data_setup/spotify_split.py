"""
This script splits the spotify data into train test and val
"""

import pandas as pd
import numpy as np

path_to_csv = '../interim/spotify_v1/spotify_v1.csv'

df = pd.read_csv(path_to_csv)

def train_test_val_split(data, test_size=0.1, val_size=0.1):
    
    np.random.seed(42)
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    data = data.iloc[indices]

    test_size = int(len(data) * test_size)
    val_size = int(len(data) * val_size)

    test = data[:test_size]

    val = data[test_size:test_size + val_size]

    train = data[test_size + val_size:]

    return train, test, val

train, test, val = train_test_val_split(df)

train.to_csv('../interim/spotify_v1/train.csv', index=False)
test.to_csv('../interim/spotify_v1/test.csv', index=False)
val.to_csv('../interim/spotify_v1/val.csv', index=False)