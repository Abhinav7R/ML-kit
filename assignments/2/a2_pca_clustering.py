"""
This script is to implement all tasks in assignment-2 questions in task 6.
Task 6
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

class PCAKMeansTasks:

    def __init__(self):
        pass

    def kmeans_with_k2():
        k=3
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)

        kmeans = KMeans(k=k)
        kmeans.fit(X)
        wcss = kmeans.getCost()

        print(f"k = {k}")
        print("WCSS: ", wcss)

    def scree_plot():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        pca = PCA()
        pca.fit(X)
        eigen_values = pca.get_eigenvalues()
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_indices]
        eigen_values = eigen_values/np.sum(eigen_values)
        threshold = 20
        eigen_values = eigen_values[:threshold]
        
        plt.plot(eigen_values)
        plt.xlabel('Number of components')
        plt.ylabel('Eigenvalues')
        plt.title('Scree plot')
        plt.grid()
        plt.xticks(range(0, threshold, 1))
        plt.savefig(f"figures/scree_plot_{threshold}.png")

    def kmeans_on_reduced_dataset():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        opt_dim = 4
        pca = PCA(n_components=opt_dim)
        pca.fit(X)
        X_pca = pca.transform(X)

        wcss = []

        k_max = 15
        for i in range(1, k_max):
            kmeans = KMeans(k=i)
            kmeans.fit(X_pca)
            wcss.append(kmeans.getCost())

        plt.plot(range(1, k_max), wcss)
        plt.title("Elbow Method on reduced dataset")
        plt.xlabel("k (Number of clusters)")
        plt.ylabel("WCSS")
        plt.xticks(range(1, k_max, 1))
        plt.grid()
        plt.savefig("figures/elbow_method_reduced_dataset.png")
            
    def kmeans3():

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

    def gmm_k2():
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

    def pca_gmm():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        opt_dim = 4
        pca = PCA(n_components=opt_dim)
        pca.fit(X)
        X_pca = pca.transform(X)
        # reduced dataset obtained
        X = X_pca

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
        plt.title("AIC and BIC for PCA + GMM on word embeddings")
        plt.legend()
        plt.savefig("figures/pca_gmm_aic_bic.png")

    def pca_gmm_sklearn():
        
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        opt_dim = 4
        pca = PCA(n_components=opt_dim)
        pca.fit(X)
        X_pca = pca.transform(X)
        # reduced dataset obtained
        X = X_pca

        aic_list = []
        bic_list = []

        print("AIC and BIC using sklearn")
        for k in range(1, 10):
            from sklearn.mixture import GaussianMixture
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
        plt.title("AIC and BIC for sklearn PCA + GMM on word embeddings")
        plt.legend()
        plt.savefig("figures/pca_gmm_aic_bic_sklearn.png")

    def gmm_k_gmm_3_reduced():
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        X = df.iloc[:, 1:].values

        opt_dim = 4
        pca = PCA(n_components=opt_dim)
        pca.fit(X)
        X_pca = pca.transform(X)
        # reduced dataset obtained
        X = X_pca

        k = 3
        print("k = ", k)
        # X = X[:,:50]
        # print("data reduced to 50 dimensions")
        print("GMM")
        gmm = GMM(k=k)
        gmm.fit(X)
        loglikelihood = gmm.getLogLikelihood()
        print("Log likelihood: ", loglikelihood/X.shape[0])
