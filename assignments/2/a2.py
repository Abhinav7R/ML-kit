"""
This script is to implement all tasks in assignment 2.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

# Task 3
from a2_kmeans import KMeansTasks

# KMeansTasks.kmeans_on_toy_dataset()
# KMeansTasks.kmeans_on_word_embeddings()
# KMeansTasks.kmeans1()


# Task 4
from a2_gmm import GMMTasks

# GMMTasks.gmm_word_embeddings()
# GMMTasks.gmm_toy_set()
# GMMTasks.gmm_aic_bic()
# GMMTasks.gmm_aic_bic_sklearn()


# Task 5
from a2_pca import PCATasks

# PCATasks.pca_on_word_embeddings()
# PCATasks.pca_using_sklearn()


# Task 6
from a2_pca_clustering import PCAKMeansTasks

# PCAKMeansTasks.kmeans_with_k2()
# PCAKMeansTasks.scree_plot()
# PCAKMeansTasks.kmeans_on_reduced_dataset()
# PCAKMeansTasks.kmeans3()
# PCAKMeansTasks.gmm_k2()
# PCAKMeansTasks.pca_gmm()
# PCAKMeansTasks.pca_gmm_sklearn()
# PCAKMeansTasks.gmm_k_gmm_3_reduced()


# Task 8
from a2_hierarchical import HierarchicalClustering

# HierarchicalClustering.hierarchical_clustering()
HierarchicalClustering.hierarchical_clusters()


# Task 9
from a2_pca_knn import PCAKNNTasks

# PCAKNNTasks.scree_plot_spotify()
# PCAKNNTasks.pca_knn_spotify()
# PCAKNNTasks.pca_knn_spotify_sklearn()