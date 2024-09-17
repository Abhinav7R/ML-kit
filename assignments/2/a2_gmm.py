"""
This script is to implement all tasks in assignment 2 question on gmm 
Task 4
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from models.gmm.gmm import GMM

class GMMTasks:

    def __init__(self):
        pass

    def gmm_word_embeddings():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        k = 3
        print("k = ", k)
        # X = X[:,:50]
        # print("data reduced to 50 dimensions")
        print("GMM")
        gmm = GMM(k=k)
        gmm.fit(X)
        loglikelihood = gmm.getLogLikelihood()
        print("Log likelihood: ", loglikelihood/X.shape[0])

        print("\nsklearn GMM")
        gmm = GaussianMixture(n_components=k, init_params='random')
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
        # hard_classes = np.argmax(classes, axis=1)
        # print("Classes: ", classes)

        colors = np.array([[1, 0, 0],  
                       [0, 1, 0],  
                       [0, 0, 1]])
        point_colours = np.dot(classes, colors)

        plt.scatter(X[:,0], X[:,1], c=point_colours)
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

    def AIC(log_likelihood, n_params):
        return 2 * n_params - 2 * log_likelihood
    
    def BIC(log_likelihood, n_params, n_samples):
        return n_params * np.log(n_samples) - 2 * log_likelihood
    
    def get_n_params(k, d):
        return k*d + k*(d*(d+1))/2 + k - 1
    
    def gmm_aic_bic():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        aic_list = []
        bic_list = []

        for k in range(1, 10):
            gmm = GMM(k=k)
            gmm.fit(X)
            loglikelihood = gmm.getLogLikelihood()
            n_params = GMMTasks.get_n_params(k, X.shape[1])
            aic = GMMTasks.AIC(loglikelihood, n_params)
            bic = GMMTasks.BIC(loglikelihood, n_params, X.shape[0])
            loglikelihood = loglikelihood/X.shape[0]
            print("k: ", k, "Log likelihood: ", loglikelihood)
            aic_list.append(aic)
            bic_list.append(bic)

        plt.plot(range(1, 10), aic_list, label="AIC")
        plt.plot(range(1, 10), bic_list, label="BIC")
        plt.xlabel("Number of clusters")
        plt.ylabel("Information Criterion")
        plt.title("AIC and BIC for GMM on word embeddings")
        plt.legend()
        plt.savefig("figures/gmm_aic_bic.png")

        # print(aic_list)
        # print(bic_list)

    def gmm_aic_bic_sklearn():

        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        aic_list = []
        bic_list = []

        print("AIC and BIC using sklearn")
        for k in range(1, 10):
            gmm = GaussianMixture(n_components=k)
            gmm.fit(X)
            loglikelihood = gmm.score(X)
            print("k: ", k, "Log likelihood: ", loglikelihood)
            aic = gmm.aic(X)
            bic = gmm.bic(X)
            # n_params = GMMTasks.get_n_params(k, X.shape[1])
            # aic = GMMTasks.AIC(loglikelihood, n_params)
            # bic = GMMTasks.BIC(loglikelihood, n_params, X.shape[0])
            aic_list.append(aic)
            bic_list.append(bic)

        plt.plot(range(1, 10), aic_list, label="AIC")
        plt.plot(range(1, 10), bic_list, label="BIC")
        plt.xlabel("Number of clusters")
        plt.ylabel("Information Criterion")
        plt.title("AIC and BIC for sklearn GMM on word embeddings")
        plt.legend()
        plt.savefig("figures/gmm_aic_bic_sklearn.png")

        print(aic_list)
        print(bic_list)

