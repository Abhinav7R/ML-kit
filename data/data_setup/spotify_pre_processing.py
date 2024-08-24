"""
This script preprcesses the data and saves it in the interim folder.
"""

import pandas as pd
import numpy as np

path_to_data = "../external/spotify.csv"
# path_to_data = "../external/spotify_2/validate.csv"

df = pd.read_csv(path_to_data, index_col=0)

df.dropna(inplace=True)

df.drop(columns=['track_id', 'track_name', 'album_name', 'artists'], inplace=True)

df['explicit'] = df['explicit'].apply(lambda x: 1 if x == True else 0)

def normalise(df):
    for column in df.columns:
        if column != 'track_genre':
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

df = normalise(df)

df.to_csv('../interim/spotify_2/validate.csv', index=False)
