"""
This script is to implement all tasks in assignment 2.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from a2_kmeans import KMeansTasks

# KMeansTasks.kmeans_on_toy_dataset()
# KMeansTasks.kmeans_on_word_embeddings()
# KMeansTasks.kmeans1()


from a2_pca import PCATasks

# PCATasks.pca_on_word_embeddings()
# PCATasks.pca_using_sklearn()


from a2_gmm import GMMTasks

# GMMTasks.gmm_word_embeddings()
GMMTasks.gmm_toy_set()