"""
This script is to implement all tasks in assignment 2 question on hierarchical clustering 
Task 8
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc

from models.pca.pca import PCA

class HierarchicalClustering:

    def __init__(self):
        pass

    def hierarchical_clustering():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values
        words = df.iloc[:, 0].values

        distance_metrics = ['euclidean', 'cityblock', 'cosine']
        linkage_methods = ['complete', 'average', 'single', 'ward', 'centroid', 'median']

        valid_combinations = [(linkage, dist) for linkage in linkage_methods for dist in distance_metrics 
                            if not (linkage in ['ward', 'centroid', 'median'] and dist != 'euclidean')]

        fig, axes = plt.subplots(len(valid_combinations) // len(distance_metrics), len(distance_metrics), figsize=(20, 15))
        axes = axes.ravel()

        for idx, (linkage, dist) in enumerate(valid_combinations):
            ax = axes[idx]
            Z = hc.linkage(X, method=linkage, metric=dist)
            dendrogram = hc.dendrogram(Z, labels=words, ax=ax)
            ax.set_title(f"{linkage.capitalize()} linkage\n{dist.capitalize()} distance")
            ax.set_xlabel("Words")
            ax.set_ylabel("Distance")
            ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.savefig("figures/hierarchical_clustering_all.png")

    def hierarchical_clusters():
        
        kbest1 = 6
        kbest2 = 3

        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values
        words = df.iloc[:, 0].values

        linkage = 'ward'
        dist = 'euclidean'
        Z = hc.linkage(X, method=linkage, metric=dist)
        print("linakge matrix")
        print(Z.shape)
        plt.figure(figsize=(10, 7))
        dendrogram = hc.dendrogram(Z, labels=words)
        plt.title(f"{linkage.capitalize()} linkage\n{dist.capitalize()} distance")
        plt.xlabel("Words")
        plt.ylabel("Distance")
        plt.xticks(rotation=90)
        plt.savefig("figures/hierarchical_clustering_clusters.png")

        clusters1 = hc.fcluster(Z, kbest1, criterion='maxclust')
        clusters2 = hc.fcluster(Z, kbest2, criterion='maxclust')

        for i in range(kbest1):
            print(f"\nCluster {i+1}:")
            words_in_cluster = words[clusters1==i+1]
            for j,word in enumerate(words_in_cluster):
                if j!=len(words_in_cluster)-1:
                    print(word, end=", ")
                else:
                    print(word)
        for i in range(kbest2):
            print(f"\nCluster {i+1}:")
            words_in_cluster = words[clusters2==i+1]
            for j,word in enumerate(words_in_cluster):
                if j!=len(words_in_cluster)-1:
                    print(word, end=", ")
                else:
                    print(word)

        #plot the clusters on 2D

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)

        colours = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
        fig, ax = plt.subplots()
        for i, x in enumerate(X_pca):
            plt.scatter(x[0], x[1], c=colours[clusters1[i]-1])
        plt.title(f"{kbest1}-clusters using Hierarchical Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/hierarchical_clustering_kbest1.png")

        fig, ax = plt.subplots()
        for i, x in enumerate(X_pca):
            plt.scatter(x[0], x[1], c=colours[clusters2[i]-1])
        plt.title(f"{kbest2}-clusters using Hierarchical Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/hierarchical_clustering_kbest2.png")




                





            
