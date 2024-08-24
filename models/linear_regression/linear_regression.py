"""
This is the linear regression model. It is implemented from scratch without using any libraries.
"""

import numpy as np
import matplotlib.pyplot as plt

# Note the code in #---------------------# is for generating images for plotting the gif

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..','..')))
from performance_measures.performance_measures import Performance_Measures

class Linear_Regression:
    # handled only dimension of X being 1
    def __init__(self, learning_rate=0.1, epochs=10000, closed_form=False, lambda_=0, regularization=None, k=1): 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.closed_form = closed_form
        self.regularization = regularization
        self.beta = None
        self.k = k

    def _fit_closed_form(self, X, y):
        if (self.regularization == 'l1' or self.regularization == 'lasso') and self.lambda_ != 0:
            raise ValueError("L1 Regularization not supported in closed form")
        else:
            self.beta = np.linalg.inv(X.T @ X + self.lambda_ * np.eye(X.shape[1])) @ X.T @ y

    def _fit_gradient_descent(self, X, y):
        m, n = X.shape
        self.beta = np.random.randn(n)

        """
        this is the loss function for linear regression
        not used in the code, but it was used for sanity check

        loss = (1 / (2 * m)) * np.sum((X @ self.beta - y)**2)
        if self.regularization == 'l1' or self.regularization == 'lasso':
            loss += self.lambda_ * np.sum(np.abs(self.beta))
        elif self.regularization == 'l2' or self.regularization == 'ridge':
            loss += self.lambda_ * np.sum(self.beta**2)
        """

        #---------------------#
        """mse_list = []
        std_list = []
        var_list = []"""
        #---------------------#

        for _ in range(self.epochs):
            gradients = (1 / m) * X.T @ (X @ self.beta - y)
            if self.regularization == 'l1' or self.regularization == 'lasso':
                gradients += (self.lambda_ / m) * np.sign(self.beta)
            elif self.regularization == 'l2' or self.regularization == 'ridge':
                gradients += (self.lambda_ / m) * self.beta
            self.beta -= self.learning_rate * gradients

            # plotting for gif
            # a plot with 4 subplots - one contains the train points scattered and the curve
            # other 3 contain the mse var and std deviation for each epoch compiled after each epoch
            
            #---------------------#
            """
            y_pred = X @ self.beta
            perf_measures = Performance_Measures()
            mse = perf_measures.mean_square_error(y, y_pred)
            std = perf_measures.standard_deviation(y, y_pred)
            var = perf_measures.variance(y, y_pred)

            epsilon = 0.001
            if _ > 10:
                if mse_list[-1] - mse < epsilon or var_list[-1] - var < epsilon or std_list[-1] - std < epsilon:
                    break

            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle('Linear Regression GIF')
            axs[0, 0].scatter(X[:, 1], y, color='blue', label='train', s=5)
            # sort the values of X[:, 1] and y_pred so that the line is continuous
            X_sorted = X[X[:, 1].argsort()]
            y_pred = X_sorted @ self.beta
            axs[0, 0].plot(X_sorted[:, 1], y_pred, color='red', label='line of fit')
            axs[0, 0].set_title('Train points and line of fit')

            mse_list.append(mse)
            std_list.append(std)
            var_list.append(var)

            axs[0, 1].plot(mse_list)
            axs[0, 1].set_title('Mean Squared Error')
            axs[0, 1].set_ylim(0, 2.2)
            axs[0, 1].set_xlim(0, self.epochs)

            axs[1, 0].plot(std_list)
            axs[1, 0].set_title('Standard Deviation')
            axs[1, 0].set_ylim(0, 1.5)
            axs[1, 0].set_xlim(0, self.epochs)

            axs[1, 1].plot(var_list)
            axs[1, 1].set_title('Variance')
            axs[1, 1].set_ylim(0, 2.2)
            axs[1, 1].set_xlim(0, self.epochs)


            #save as 4 digit number with leading zeros
            save_As = str(_).zfill(4)
            plt.savefig(f'../../assignments/1/figures/gif_figures/{self.k}/{self.k}_{save_As}.png')

            plt.close()
            """
        #---------------------#
            
            
    def fit(self, X, y):
        polynomial_degree = self.k
        for i in range(2, polynomial_degree + 1):
            X = np.c_[X, X[:, 0]**i]
        X = np.c_[np.ones(X.shape[0]), X]
        if self.closed_form:
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)

    def predict(self, X):
        polynomial_degree = self.k
        for i in range(2, polynomial_degree + 1):
            X = np.c_[X, X[:, 0]**i]
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.beta
            
    def get_params(self):
        return self.beta
      