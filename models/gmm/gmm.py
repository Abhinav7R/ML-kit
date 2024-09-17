"""
This is the GMM - Gaussian Mixture Model. It is implemented from scratch without using any libraries.
"""

import numpy as np
import scipy
import copy
from scipy.stats import multivariate_normal

class GMM:

    def __init__(self, k=3, max_iters=300):
        self.n_clusters = k
        self.max_iters = max_iters
        self.means = None
        self.covariances = None
        self.priors = None
        self.responsibilities = None

    def _init_params(self):

        random_sample_idxs = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.means = self.X[random_sample_idxs]

        self.covariances = np.zeros((self.n_clusters, self.n_features, self.n_features))

        # for i in range(self.n_clusters):
        #     A = np.random.random((self.n_features, self.n_features))
        #     self.covariances[i] = np.dot(A, A.T) + 1e-6 * np.eye(self.n_features)

        for i in range(self.n_clusters):
            self.covariances[i] = np.eye(self.n_features)

        # prior probabilities
        self.priors = np.random.random(self.n_clusters)
        self.priors /= np.sum(self.priors)


    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self._init_params()
        
        for iteration in range(self.max_iters):
            
            prev_log_likelihood = self.getLogLikelihood()
            prev_means = copy.deepcopy(self.means)
            prev_covariances = copy.deepcopy(self.covariances)
            prev_priors = copy.deepcopy(self.priors)
            prev_responsibilities = copy.deepcopy(self.responsibilities)

            # print("Log likelihood: ", prev_log_likelihood)
            
            self._e_step()
            self._m_step()

            log_likelihood = self.getLogLikelihood()

            if np.abs(log_likelihood - prev_log_likelihood) < 1e-6:
                break

            if log_likelihood < prev_log_likelihood:
                # print("Log likelihood decreased. Exiting...")
                self.means = prev_means
                self.covariances = prev_covariances
                self.priors = prev_priors
                self.responsibilities = prev_responsibilities
                break

    def _e_step(self):
        # Expectation step
        self.responsibilities = np.zeros((self.n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            self.responsibilities[:, j] = self.priors[j] * multivariate_normal.pdf(
                self.X, mean=self.means[j], cov=self.covariances[j], allow_singular=True)
            
        responsibility_sums = np.sum(self.responsibilities, axis=1, keepdims=True)
        responsibility_sums = np.where(responsibility_sums == 0, 1, responsibility_sums)
        self.responsibilities = self.responsibilities / responsibility_sums

    def _m_step(self):
        # Maximization step
        for j in range(self.n_clusters):
            Nk = np.sum(self.responsibilities[:, j])
            self.means[j] = np.sum(self.responsibilities[:, j].reshape(-1, 1) * self.X, axis=0) / Nk
            self.covariances[j] = np.dot((self.responsibilities[:, j].reshape(-1, 1) * (self.X - self.means[j])).T, (self.X - self.means[j])) / Nk
            self.priors[j] = Nk / self.n_samples

    def getParams(self):
        return self.means, self.covariances, self.priors

    def getMembership(self):
        return self.responsibilities
    
    def getLikelihood(self):
        likelihoods = np.zeros((self.n_samples, self.n_clusters))

        for j in range(self.n_clusters):
            likelihoods[:, j] = self.priors[j] * multivariate_normal.pdf(
                self.X, mean=self.means[j], cov=self.covariances[j], allow_singular=True)
            
        total_likelihood = np.sum(likelihoods, axis=1)
        overall_likelihood = np.prod(total_likelihood + 1e-10)  

        return overall_likelihood
    
    def getLogLikelihood(self):       
        likelihoods = np.zeros((self.n_samples, self.n_clusters))

        for j in range(self.n_clusters):
            likelihoods[:, j]  = self.priors[j] * multivariate_normal.pdf(
                self.X, mean=self.means[j], cov=self.covariances[j], allow_singular=True)
        
        total_likelihood = np.sum(likelihoods, axis=1)
        
        log_likelihood = np.sum(np.log(total_likelihood))

        return log_likelihood
    