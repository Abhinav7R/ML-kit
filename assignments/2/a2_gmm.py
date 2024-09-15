"""
This script is to implement all tasks in assignment 2 question on gmm.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from models.gmm.gmm import GMM

class GMMTasks:

    def __init__(self):
        pass

    def gmm_word_embeddings():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        # X = X[:,:80]

        gmm = GMM(k=3)
        gmm.fit(X)
        loglikelihood = gmm.getLogLikelihood()
        print("Log likelihood: ", loglikelihood/X.shape[0])

        print("\nsklearn GMM")
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3, init_params='random')
        gmm.fit(X)
        loglikelihood = gmm.score(X)
        print("Log likelihood: ", loglikelihood)

    def gmm_toy_set():
        path_to_toy_dataset = "../../data/external/kmeans_toy_data.csv"
        df = pd.read_csv(path_to_toy_dataset)
        x = df['x'].values
        y = df['y'].values
        X = np.array(list(zip(x, y)))

        gmm = GMM(k=3)
        gmm.fit(X)
        loglikelihood = gmm.getLogLikelihood()
        print("final Log likelihood: ", loglikelihood/X.shape[0])
        classes = gmm.getMembership()
        hard_classes = np.argmax(classes, axis=1)
        # print("Classes: ", classes)

        plt.scatter(X[:,0], X[:,1], c=classes)
        plt.title("GMM on toy dataset")
        plt.savefig("figures/gmm_toy_dataset.png")

        
        print("sklearn GMM")
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3)
        gmm.fit(X)
        classes = gmm.predict(X)
        # print("Classes: ", classes)
        loglikelihood = gmm.score(X)
        print("final Log likelihood: ", loglikelihood)

        plt.scatter(X[:,0], X[:,1], c=classes)
        plt.title("sklearn GMM on toy dataset")
        plt.savefig("figures/gmm_toy_dataset_sklearn.png")


        
