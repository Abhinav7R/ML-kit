"""
This is the K-Means model. It is implemented from scratch without using any libraries.
"""

import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=300):
        self.n_clusters = k
        self.max_iters = max_iters
        self.X = None
        self.n_samples = None
        self.n_features = None
        self.centroids = None

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        random_sample_idxs = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.centroids = self.X[random_sample_idxs]
        self.cluster_labels = np.zeros(self.n_samples)
        self._train()

    def _train(self):
        for _ in range(self.max_iters):
            for i, x in enumerate(self.X):
                distances = np.linalg.norm(self.centroids - x, axis=1)
                self.cluster_labels[i] = np.argmin(distances)
            new_centroids = self._find_centroids()
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _find_centroids(self):
        centroids = np.zeros((self.n_clusters, self.n_features))
        for cluster in range(self.n_clusters):
            cluster_points = self.X[self.cluster_labels == cluster]
            centroids[cluster] = np.mean(cluster_points, axis=0)
        return centroids
    
    def predict(self, X):
        cluster_labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.centroids - x, axis=1)
            cluster_labels[i] = np.argmin(distances)
        return cluster_labels
    
    def getCost(self):
        # within-cluster sum of squares
        cost = 0
        for cluster in range(self.n_clusters):
            cluster_points = self.X[self.cluster_labels == cluster]
            cost += np.sum((cluster_points - self.centroids[cluster])**2)
        return cost
    
    def get_centroids(self):
        return self.centroids
    
