"""
This is the PCA model. It is implemented from scratch without using any libraries.
"""

import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.X = None
        self.n_samples = None
        self.n_features = None
        self.mean = None
        self.covariance_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.projection_matrix = None

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.mean = np.mean(X, axis=0)
        X_centered = self.X - self.mean
        self.covariance_matrix = np.cov(X_centered.T)
        # print(self.covariance_matrix.shape)
        # eigh for not having complex numbers
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance_matrix)
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.sorted_eigenvectors = self.eigenvectors[:, sorted_indices]
        self.projection_matrix = self.sorted_eigenvectors[:, :self.n_components]
        # print(self.projection_matrix.shape)

    def transform(self, X):
        X_centered = X - self.mean
        self.X_pca = X_centered.dot(self.projection_matrix)
        # print(self.X_pca.shape)
        
        return self.X_pca
    
    def inverse_transform(self, X_pca):
        return X_pca.dot(self.projection_matrix.T) + self.mean

    
    def checkPCA(self):
        if self.X_pca is None:
           raise Exception("PCA not done yet. Please run fit method first.")
        if self.X_pca.shape[1] != self.n_components:
           return False
        
        X_reconstructed = self.inverse_transform(self.X_pca)

        reconstruction_error = np.mean((self.X - X_reconstructed) ** 2)

        print("Reconstruction error: ", reconstruction_error)

        if reconstruction_error > 1e-1:
            return False
        
        return True
       

       
    