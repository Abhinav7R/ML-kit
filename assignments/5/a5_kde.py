"""
This script is for Asg 5 KDE 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.KDE.kde import KDE
from models.gmm.gmm import GMM


def generate_circle_data():
    n_big = 3000
    r_big = 2
    theta_big = np.random.uniform(0, 2 * np.pi, n_big)
    radii_large = np.sqrt(np.random.uniform(0, r_big**2, n_big))
    x_large = radii_large * np.cos(theta_big)
    y_large = radii_large * np.sin(theta_big)

    noise = np.random.uniform(-1, 1, 2*n_big) * 0.25
    x_large += noise[:n_big]
    y_large += noise[n_big:]

    n_small = 500
    r_small = 0.25
    theta_small = np.random.uniform(0, 2 * np.pi, n_small)
    radii_small = np.sqrt(np.random.uniform(0, r_small**2, n_small))
    x_small = radii_small * np.cos(theta_small) + 1
    y_small = radii_small * np.sin(theta_small) + 1

    x = np.concatenate((x_large, x_small))
    y = np.concatenate((y_large, y_small))

    data = np.vstack((x, y)).T

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=0.5, color='black')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Original Data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    return data

def kde_on_circle_data(data):
    kde = KDE(h=0.3, kernel='gaussian')
    kde.fit(data)
    p_kde = kde.predict()
    kde.plot()

def sklearn_hmm(data):
    from sklearn.neighbors import KernelDensity
    kde_sklearn = KernelDensity(bandwidth=0.3, kernel='gaussian')
    kde_sklearn.fit(data)

    dens_sklearn = np.exp(kde_sklearn.score_samples(data))

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=dens_sklearn, s=0.75, cmap='viridis')
    plt.colorbar()
    plt.title("Sklearn KDE plot on data points with Gaussian kernel and h = 0.3")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def different_h_kernels(data):
    x = data[:, 0]
    y = data[:, 1]
    h = [0.1, 0.3, 0.5]
    kernels = ['box', 'gaussian', 'triangular']

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):
            kde = KDE(h=h[i], kernel=kernels[j])
            kde.fit(data)
            p_kde = kde.predict()
            axs[i, j].scatter(x, y, c=p_kde, s=0.5, cmap='viridis')
            axs[i, j].set_title(f"Kernel: {kernels[j]}, Bandwidth: {h[i]}")
            axs[i, j].set_xlabel("X-axis")
            axs[i, j].set_ylabel("Y-axis")
            # axs[i, j].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def gmm_on_circle_data(data, k=2):
    x = data[:, 0]
    y = data[:, 1]
    if k > 6:
        print("Can't have more than 6 colours for plotting")
        return
    gmm = GMM(k=k, max_iters=100)
    gmm.fit(data)
    p_gmm = gmm.getMembership()

    colors = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ])
    point_colours = np.dot(p_gmm, colors[:k])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=point_colours, s=0.5)
    plt.title("GMM")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def get_subplots_gmm_for_diff_k(data, k_values):
    x = data[:, 0]
    y = data[:, 1]
    fig, axs = plt.subplots(1, len(k_values), figsize=(20, 4))
    for i in range(len(k_values)):
        k = k_values[i]
        gmm = GMM(k=k, max_iters=100)
        gmm.fit(data)
        p_gmm = gmm.getMembership()

        colors = np.array([
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ])
        point_colours = np.dot(p_gmm, colors[:k])

        axs[i].set_title(f"GMM with k = {k}")
        axs[i].scatter(x, y, c=point_colours, s=0.5)
    plt.tight_layout()
    plt.show()


# data = generate_circle_data()
# kde_on_circle_data(data)
# sklearn_hmm(data)
# different_h_kernels(data)
# gmm_on_circle_data(data, k=2)
# get_subplots_gmm_for_diff_k(data, [2, 3, 4, 5, 6])