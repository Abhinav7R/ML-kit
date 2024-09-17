"""
This script is to implement all tasks in assignment-2 questions in task 9.
Task 9
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.pca.pca import PCA
from models.knn.knn import OptimisedKNN as KNN
from performance_measures.performance_measures import Performance_Measures

from time import time

class PCAKNNTasks:

    def __init__(self):
        pass
        
    def get_spotify_numerical_only():
        #this is without normalisation
        path_to_spotify = "../../data/external/spotify.csv"
        df = pd.read_csv(path_to_spotify, index_col=0)
        df.dropna(inplace=True)
        df.drop(columns=['track_id', 'track_name', 'album_name', 'artists'], inplace=True)
        df['explicit'] = df['explicit'].apply(lambda x: 1 if x == True else 0)

        with open("../../data/interim/spotify_v1/spotify_numerical_only.csv", 'w') as f:
            df.to_csv(f)

    def scree_plot_spotify():
        path_to_spotify = "../../data/interim/spotify_v1/spotify_numerical_only.csv"
        df = pd.read_csv(path_to_spotify, index_col=0)
        X = df.drop(columns=['track_genre']).values

        pca = PCA()
        pca.fit(X)
        eigen_values = pca.get_eigenvalues()
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_indices]
        eigen_values = eigen_values/np.sum(eigen_values)
        threshold = 15
        eigen_values = eigen_values[:threshold]
        
        plt.plot(eigen_values)
        plt.xlabel('Number of components')
        plt.ylabel('Eigenvalues')
        plt.title('Scree plot on spotify data')
        plt.grid()
        plt.yscale('log')
        plt.xticks(range(0, threshold, 1))
        plt.savefig(f"figures/spotify_scree_plot_{threshold}.png")

    def train_test_val_split(data, test_size=0.1, val_size=0.1):
    
        np.random.seed(42)
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        data = data.iloc[indices]

        test_size = int(len(data) * test_size)
        val_size = int(len(data) * val_size)

        test = data[:test_size]

        val = data[test_size:test_size + val_size]

        train = data[test_size + val_size:]

        return train, test, val

    def pca_knn_spotify():
        # path_to_spotify = "../../data/interim/spotify_v1/spotify_numerical_only.csv"
        # df = pd.read_csv(path_to_spotify, index_col=0)
        # X = df.drop(columns=['track_genre']).values

        # n_comp = 6
        # pca = PCA(n_components=n_comp)
        # pca.fit(X)
        # X_pca = pca.transform(X)
        # print(f"PCA to {n_comp} components")
        # if pca.checkPCA():
        #     print("PCA check: True")

        # X_pca = pd.DataFrame(X_pca)
        # X_pca['track_genre'] = df['track_genre'].values

        # print(X_pca.head())

        path_to_spotify = "../../data/interim/spotify_v1/spotify_reduced.csv"
        X_pca = pd.read_csv(path_to_spotify, index_col=0)

        train, test, val = PCAKNNTasks.train_test_val_split(X_pca)

        X_train = train.drop(columns=['track_genre']).values
        y_train = train['track_genre'].values

        X_val = val.drop(columns=['track_genre']).values
        y_val = val['track_genre'].values

        start_time = time()
        knn = KNN(k=20, distance_metric='manhattan')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        end_time = time()

        performance = Performance_Measures()
        accuracy = performance.accuracy(y_val, y_pred)
        print(f"Accuracy: {accuracy}")

        precision = performance.precision(y_val, y_pred)
        print(f"Precision: {precision}")

        recall = performance.recall(y_val, y_pred)
        print(f"Recall: {recall}")

        f1 = performance.f1_score(y_val, y_pred)
        print(f"F1 score: {f1}")

        print(f"Time taken: {end_time - start_time}")
        print(f"Time taken with perf measures: ", time() - start_time)
        
    def pca_knn_spotify_sklearn():
        
        path_to_spotify = "../../data/interim/spotify_v1/spotify_numerical_only.csv"
        df = pd.read_csv(path_to_spotify, index_col=0)
        X = df.drop(columns=['track_genre']).values

        n_comp = 6
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)
        print(f"PCA to {n_comp} components")
        if pca.checkPCA():
            print("PCA check: True")

        X_pca = pd.DataFrame(X_pca)
        X_pca['track_genre'] = df['track_genre'].values

        train, test, val = PCAKNNTasks.train_test_val_split(df)

        X_train = train.drop(columns=['track_genre']).values
        y_train = train['track_genre'].values

        X_val = val.drop(columns=['track_genre']).values
        y_val = val['track_genre'].values

        from sklearn.neighbors import KNeighborsClassifier
        start_time = time()
        knn = KNeighborsClassifier(n_neighbors=20, metric='manhattan')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        end_time = time()

        performance = Performance_Measures()
        accuracy = performance.accuracy(y_val, y_pred)
        print(f"Accuracy: {accuracy}")

        precision = performance.precision(y_val, y_pred)
        print(f"Precision: {precision}")

        recall = performance.recall(y_val, y_pred)
        print(f"Recall: {recall}")

        f1 = performance.f1_score(y_val, y_pred)
        print(f"F1 score: {f1}")

        print(f"Time taken: {end_time - start_time}")
        print(f"Time taken with perf measures: ", time() - start_time)
        
    def plot_times():
        times = [113, 81]
        labels = ['KNN', 'KNN with PCA']
        plt.bar(labels, times)
        plt.ylabel('Time taken in seconds')
        plt.title('Time taken for KNN and KNN with PCA')
        plt.savefig('figures/knn_pca_time.png')

