"""
Kernel Density Estimation for d dimensional data (n points) from scratch for different kernels
"""

import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(self, h, kernel):
        self.data = None
        self.h = h
        self.kernel = kernel
        self.n = None
        self.d = None
        self.kde = None

    def K(self, x):
        # box, gaussian and traingular kernels
        if self.kernel == 'box':
            return np.prod(np.abs(x) <= 0.5, axis=1)
        elif self.kernel == 'gaussian':
            return np.exp(-0.5*np.sum(x**2, axis=1)) * (2*np.pi)**(-self.d/2)
        elif self.kernel == 'triangular':
            return np.maximum(1 - np.sum(np.abs(x), axis=1), 0)
        else:
            print("Invalid kernel")
            return None

    def fit(self, data):
        self.data = data
        self.n, self.d = data.shape
        
    
    def predict(self):
        self.kde = np.sum([self.K((self.data - self.data[i])/self.h) for i in range(self.n)], axis=0)/(self.n*self.h**self.d)
        return self.kde
    
    def plot(self):
        if self.d == 2:
            x = self.data[:, 0]
            y = self.data[:, 1]
            colour = self.predict()
            plt.figure()
            plt.scatter(x, y, c=colour, s=0.75, cmap='viridis')
            plt.colorbar()
            plt.title("KDE plot: " + self.kernel + " kernel" + " with h = " + str(self.h))
            plt.xlabel("X")
            plt.ylabel("Y")
            # plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()

        else:
            print("Can't plot for other than 2D data")

    
    