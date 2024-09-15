"""
This script is to implement all tasks in assignment 2 question on pca.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.pca.pca import PCA

class PCATasks:

    def __init__(self):
        pass

    def pca_on_word_embeddings():
        print("PCA on word embeddings\n")
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)
        # print("Shape of transformed data",X_pca.shape)
        print("PCA to 2 components")
        if pca.checkPCA():
            print("PCA check: True")

        # print(X_pca)

        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.title("PCA on word embeddings, n_components=2")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/pca_2.png")
        plt.close()

        pca = PCA(n_components=3)
        pca.fit(X)
        X_pca = pca.transform(X)
        # print("Shape of transformed data",X_pca.shape)
        print("PCA to 3 components")
        if pca.checkPCA():
            print("PCA check: True")

        # print(X_pca)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
        ax.set_title("PCA on word embeddings, n_components=3")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        plt.savefig("figures/pca_3.png")
        plt.close()

    def pca_using_sklearn():
        print("\nPCA using sklearn\n")
        path_to_word_embeddings = "../../data/processed/word_embeddings/word_embeddings.csv"
        df = pd.read_csv(path_to_word_embeddings)
        # print(df.head())
        X = df.iloc[:, 1:].values
        # print(X.shape)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        # print("Shape of transformed data",X_pca.shape)

        print("PCA to 2 components")
        reconstruction_error = np.mean((X - pca.inverse_transform(X_pca)) ** 2)
        print("Reconstruction error: ", reconstruction_error)

        # print(X_pca)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='orange')
        plt.title("PCA on word embeddings using sklearn, n_components=2")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("figures/pca_2_sklearn.png")
        plt.close()

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        # print("Shape of transformed data",X_pca.shape)

        print("PCA to 3 components")
        reconstruction_error = np.mean((X - pca.inverse_transform(X_pca)) ** 2)
        print("Reconstruction error: ", reconstruction_error)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color='orange')
        ax.set_title("PCA on word embeddings using sklearn, n_components=3")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        plt.savefig("figures/pca_3_sklearn.png")
        plt.close()





        
    
