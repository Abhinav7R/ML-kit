"""
Visualising the spotify data for the KNN model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_to_data = "../../../data/interim/spotify_v1/spotify_v1.csv"
path_to_figures = "../figures/knn_spotify/data_vis"

df = pd.read_csv(path_to_data)

def pie_chart_for_genres():
    genre = df['track_genre']
    genre_counts = genre.value_counts().nlargest(40)
    plt.figure(figsize=(10, 10))
    plt.pie(genre_counts, labels=genre_counts.index)
    plt.title("Distribution of genres in the dataset")
    plt.savefig(f"{path_to_figures}/genre_distribution.png")
    plt.close()

def histograms_for_features():
    fig, axs = plt.subplots(3, 5, figsize=(20, 20))
    axs = axs.ravel()
    for column in df.columns:
        if column != 'track_genre':
            axs[df.columns.get_loc(column)].hist(df[column])
            axs[df.columns.get_loc(column)].set_title(column)
    plt.savefig(f"{path_to_figures}/feature_histograms.png")

def violin_plot():
    fig, axs = plt.subplots(3, 5, figsize=(20, 20))
    axs = axs.ravel()
    for i, column in enumerate(df.columns):
        if column != 'track_genre':
            axs[i].violinplot(df[column])
            axs[i].set_title(column)
    plt.savefig(f"{path_to_figures}/feature_violin_plots.png")

def correlation_matrix():
    genre_map = {}
    for i in range(len(df)):
        genre = df['track_genre'].iloc[i]
        if genre not in genre_map:
            genre_map[genre] = len(genre_map) + 1

    df['track_genre'] = df['track_genre'].apply(lambda x: genre_map[x])

    corr = df.corr()
    plt.figure(figsize=(20, 20))
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.savefig(f"{path_to_figures}/correlation.png")

    #print the first 10 most correlated features with track_genre
    print(corr['track_genre'].nlargest(10))


# pie_chart_for_genres()
# histograms_for_features()
# violin_plot()
# correlation_matrix()




