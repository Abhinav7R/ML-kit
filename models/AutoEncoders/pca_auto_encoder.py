"""
This is the PCA Auto Encoder
"""

import numpy as np

class PcaAutoencoder:
    def __init__(self, num_components=None):
        if num_components == None:
            raise ValueError("Provide number of components")
        self.num_components = num_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X):
        """
        Calculates the mean, eigenvalues, and eigenvectors of the input data.
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.eigenvalues = eigenvalues[sorted_indices]
        
        self.eigenvectors = self.eigenvectors[:, :self.num_components]

    def encode(self, X):
        """
        Reduces the dimensionality of the input data using learned eigenvectors.
        """
        X_centered = X - self.mean
        encoded = np.dot(X_centered, self.eigenvectors)
        return encoded

    def forward(self, X):
        """
        Reconstructs the data from the reduced representation.
        """
        encoded = self.encode(X)
        reconstructed = np.dot(encoded, self.eigenvectors.T) + self.mean
        return reconstructed

    def reconstruction_loss(self, X, X_reconstructed):
        """
        Calculates the mean squared error between the original and reconstructed data.
        """
        return np.mean((X - X_reconstructed) ** 2)
