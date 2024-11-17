"""
Hidden Markov Model (for digit recognition)
using GaussianHMM from hmmlearn library
"""

import numpy as np
from hmmlearn import hmm

class HMM:
    def __init__(self, num_of_digits=10):
        self.models = {}
        self.num_of_digits = num_of_digits

    def train(self, train_data, n_components=10, n_iter=100):
        for digit, mfcc_list in train_data.items():
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
            X = np.concatenate(mfcc_list)
            model.fit(X)
            self.models[digit] = model

    def predict(self, test_data):
        preds = []
        for mcff in test_data:
            scores = [model.score(mcff) for model in self.models.values()]
            pred = np.argmax(scores)
            preds.append(pred)
        return preds
    
