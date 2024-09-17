"""
This script is to implement all tasks in assignment 2 question on kmeans.
Task 3
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.k_means.k_means import KMeans

class KMeansTasks:

    def __init__(self):
        pass

    def kmeans_on_toy_dataset():
        path_to_toy_dataset = "../../data/external/kmeans_toy_data.csv"
        df = pd.read_csv(path_to_toy_dataset)
        # print(df.head())
        x = df['x'].values
        y = df['y'].values
        colours = {0: 'r', 1: 'g', 2: 'b'}

        plt.scatter(x, y, c='black')
        plt.title("Toy dataset")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.savefig("figures/toy_dataset.png")

        X = np.array(list(zip(x, y)))
        # print(X.shape)
        kmeans = KMeans(k=3)
        kmeans.fit(X)
        cluster_labels = kmeans.predict(X)

        for i, x in enumerate(X):
            plt.scatter(x[0], x[1], c=colours[cluster_labels[i]])
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', marker='x')
        plt.title("K-Means clustering on toy dataset")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.savefig("figures/kmeans_toy_dataset.png")

        print("WCSS: ", kmeans.getCost().round(2))

    def kmeans_on_word_embeddings():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)

        wcss = []
        k_max = 10
        for i in range(1, k_max):
            kmeans = KMeans(k=i)
            kmeans.fit(X)
            wcss.append(kmeans.getCost())

        plt.plot(range(1, k_max), wcss)
        plt.title("Elbow Method")
        plt.xlabel("k (Number of clusters)")
        plt.ylabel("WCSS")
        # plt.savefig("figures/elbow_method.png")

        print("WCSS: ", wcss)

    def kmeans1():

        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)

        kmeans = KMeans(k=6)
        kmeans.fit(X)
        wcss = kmeans.getCost()

        print("k = 6")
        print("WCSS: ", wcss)
