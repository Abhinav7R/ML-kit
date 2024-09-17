"""
This script is to implement all tasks in assignment-2 questions in task 7.
Task 7
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.pca.pca import PCA
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM

from a2_gmm import GMMTasks

class ClusterAnalysisTasks:

    def __init__(self):
        self.kmeans1 = 6
        self.k2 = 3
        self.kmeans3 = 6

        self.kgmm1 = 1
        self.kgmm3 = 3

    def get_data_in_2D():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)

        return X_pca
    
    def get_data_in_4D():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        pca = PCA(n_components=4)
        pca.fit(X)
        X_pca = pca.transform(X)

        return X_pca

    def analysis_kmeans1(self):
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)
        words = df.iloc[:, 0].values

        kmeans = KMeans(k=self.kmeans1)
        kmeans.fit(X)
        wcss = kmeans.getCost()

        print(f"k = {self.kmeans1}")
        print("WCSS: ", wcss)

        #plot the clusters
        X_pca = ClusterAnalysisTasks.get_data_in_2D()
        cluster_labels = kmeans.predict(X)
        # print(cluster_labels)
        colours = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
        for i, x in enumerate(X_pca):
            plt.scatter(x[0], x[1], c=colours[cluster_labels[i]])
        plt.title(f"K-Means clustering on word embeddings, k={self.kmeans1}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/analysis_kmeans1.png")

        # print the words in each cluster
        for i in range(self.kmeans1):
            print(f"\n\nCluster {i+1}:")
            words_in_cluster = words[cluster_labels==i]
            for j,word in enumerate(words_in_cluster):
                if j!=len(words_in_cluster)-1:
                    print(word, end=", ")
                else:
                    print(word)
        print("\n")

    def analysis_kmeans3(self):
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)
        words = df.iloc[:, 0].values

        X_pca_4 = ClusterAnalysisTasks.get_data_in_4D()
        kmeans = KMeans(k=self.kmeans3)
        kmeans.fit(X_pca_4)
        wcss = kmeans.getCost()
        cluster_labels = kmeans.predict(X_pca_4)

        print(f"k = {self.kmeans3}")
        print("WCSS: ", wcss)

        #plot the clusters
        X_pca = ClusterAnalysisTasks.get_data_in_2D()
        # print(cluster_labels)
        colours = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
        for i, x in enumerate(X_pca):
            plt.scatter(x[0], x[1], c=colours[cluster_labels[i]])
        plt.title(f"K-Means clustering on word embeddings, k={self.kmeans3}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/analysis_kmeans3.png")

        # print the words in each cluster
        for i in range(self.kmeans3):
            print(f"\nCluster {i+1}:")
            words_in_cluster = words[cluster_labels==i]
            for j,word in enumerate(words_in_cluster):
                if j!=len(words_in_cluster)-1:
                    print(word, end=", ")
                else:
                    print(word)
        print("\n")
    
    def visual_analysis_k2(self):
        #PCA to 2 components
        # write the words in each cluster on the image
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)
        words = df.iloc[:, 0].values

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)

        #plot the clusters with words

        x= X_pca[:,0]
        y= X_pca[:,1]

        plt.figure(figsize=(25, 20))
        plt.scatter(x, y, c='black')
        for i, word in enumerate(words):
            plt.text(x[i] + 0.01, y[i] + 0.01, word, fontsize=9, alpha=0.75)   
        plt.title("PCA to 2 components and words")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/visual_analysis.png") 
        
    def analysis_gmm3(self):
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)
        words = df.iloc[:, 0].values

        X_pca_4 = ClusterAnalysisTasks.get_data_in_4D()
        gmm = GMM(k=self.kgmm3)
        gmm.fit(X_pca_4)
        loglikelihood = gmm.getLogLikelihood()

        print(f"k = {self.kgmm3}")
        print("Log likelihood: ", loglikelihood/X.shape[0])

        responsibilities = gmm.getMembership()
        cluster_labels = np.argmax(responsibilities, axis=1)

        #plot the clusters
        X_pca = ClusterAnalysisTasks.get_data_in_2D()
        # print(cluster_labels)
        colours = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
        for i, x in enumerate(X_pca):
            plt.scatter(x[0], x[1], c=colours[cluster_labels[i]])
        plt.title(f"GMM clustering on word embeddings, k={self.kgmm3}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/analysis_kgmm3.png")

        # print the words in each cluster
        for i in range(self.kgmm3):
            print(f"\nCluster {i+1}:")
            words_in_cluster = words[cluster_labels==i]
            for j,word in enumerate(words_in_cluster):
                if j!=len(words_in_cluster)-1:
                    print(word, end=", ")
                else:
                    print(word)
        print("\n")


