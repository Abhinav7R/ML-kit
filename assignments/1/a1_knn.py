"""
This script is the main script that uses the KNN model to predict the genre of the tracks.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import pandas as pd
import time
import re
import matplotlib.pyplot as plt
import numpy as np

from models.knn.knn import initialKNN, OptimisedKNN
from performance_measures.performance_measures import Performance_Measures

def hyperparameter_tuning():

    train = pd.read_csv('../../data/interim/spotify_v1/train.csv')
    val = pd.read_csv('../../data/interim/spotify_v1/val.csv')

    X_train = train.drop(columns=['track_genre']).values
    y_train = train['track_genre'].values

    X_val = val.drop(columns=['track_genre']).values
    y_val = val['track_genre'].values

    def knn_experiment(k, distance_metric):
        time_start = time.time()
        knn1 = OptimisedKNN(k, distance_metric)

        knn1.fit(X_train, y_train)

        y_pred = knn1.predict(X_val)

        # performance measures

        perf_measures = Performance_Measures()

        acc = perf_measures.accuracy(y_val, y_pred)
        prec = perf_measures.precision(y_val, y_pred)
        rec = perf_measures.recall(y_val, y_pred)
        f1 = perf_measures.f1_score(y_val, y_pred)

        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}")

        time_end = time.time()

        with open("knn_results.txt", "a") as f:
            f.write(f"K: {k}, Distance Metric: {distance_metric}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Precision: {prec}\n")
            f.write(f"Recall: {rec}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Time taken: {time_end - time_start}\n")
            f.write("\n")

    ks = [i for i in range(20, 36)]
    distance_metrics = ['euclidean', 'cosine', 'manhattan']


    for k in ks:
        for distance_metric in distance_metrics:
            print(f"K: {k}, Distance Metric: {distance_metric}")
            knn_experiment(k, distance_metric)
            print("\n")

def time_for_pred_based_on_train_size():

    val = pd.read_csv('../../data/interim/spotify_v1/val.csv')
    train_dataset_size = [1000, 2000, 5000, 10000, 20000, 50000, 90000]

    for size in train_dataset_size:
        start_time = time.time()
        train = pd.read_csv('../../data/interim/spotify_v1/train.csv')
        train = train[:size]
        print(f"Train size: {size}")
        X_train = train.drop(columns=['track_genre']).values
        y_train = train['track_genre'].values

        X_val = val.drop(columns=['track_genre']).values
        y_val = val['track_genre'].values

        from sklearn.neighbors import KNeighborsClassifier
        knn1 = KNeighborsClassifier(n_neighbors=20, metric='manhattan')

        knn1.fit(X_train, y_train)

        y_pred = knn1.predict(X_val)

        # performance measures

        perf_measures = Performance_Measures()

        acc = perf_measures.accuracy(y_val, y_pred)
        prec = perf_measures.precision(y_val, y_pred)
        rec = perf_measures.recall(y_val, y_pred)
        f1 = perf_measures.f1_score(y_val, y_pred)

        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}")
        print("\n")

        end_time = time.time()

        with open("knn_results_time.txt", "a") as f:
            f.write("Sklearn KNN\n")
            f.write(f"Train size: {size}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Precision: {prec}\n")
            f.write(f"Recall: {rec}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Time taken: {end_time - start_time}\n")
            f.write("\n")


def parse_results_k_dist(file_path):
    
    best_k = None
    best_metric = None
    best_accuracy = 0
    metrics_data = {'euclidean': [], 'cosine': [], 'manhattan': []}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('K:'):
                k_match = re.search(r'K: (\d+)', line)
                metric_match = re.search(r'Distance Metric: (\w+)', line)
                if k_match and metric_match:
                    k = int(k_match.group(1))
                    metric = metric_match.group(1)

            if line.startswith('Accuracy:'):
                accuracy = float(line.split(': ')[1])

                if metric in metrics_data:
                    metrics_data[metric].append((k, accuracy))
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k
                    best_metric = metric
    print("Best Accuracy: ",best_accuracy)
    print("Best K: ",best_k)
    print("Best Metric: ",best_metric)
    return metrics_data

def plot_results_k_dist(metrics_data):
    
    plt.figure(figsize=(12, 8))

    for metric, data in metrics_data.items():
        data.sort()
        ks, accuracies = zip(*data)
        plt.plot(ks, accuracies, marker='o', label=f'{metric.capitalize()}')

    plt.title('Accuracy vs K for Different Distance Metrics')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/knn_spotify/best_k_distance_metric.png')

def best_k_distance_metric():
    file_path = 'results/knn_results.txt'
    metrics_data = parse_results_k_dist(file_path)
    plot_results_k_dist(metrics_data)


def parse_knn_time_results(file_path):
    results = {
        'Initial KNN': [],
        'Optimised KNN': [],
        'Sklearn KNN': []
    }
    train_sizes = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'Initial KNN' in line:
                model_type = 'Initial KNN'
            elif 'Optimised KNN' in line:
                model_type = 'Optimised KNN'
            elif 'Sklearn KNN' in line:
                model_type = 'Sklearn KNN'

            train_size_match = re.search(r'Train size: (\d+)', line)
            if train_size_match:
                train_size = int(train_size_match.group(1))
                if train_size not in train_sizes:
                    train_sizes.append(train_size)

            time_match = re.search(r'Time taken: ([\d\.]+)', line)
            if time_match:
                time_taken = float(time_match.group(1))
                results[model_type].append(time_taken)

    return train_sizes, results

def plot_time_vs_train_size(train_sizes, results):
    plt.figure(figsize=(10, 6))

    for model_type, times in results.items():
        plt.plot(train_sizes, times, marker='o', markersize=4, label=model_type)

    plt.title('Time Taken vs Train Set Size for Different KNN Models')
    plt.xlabel('Train Set Size')
    plt.ylabel('Time Taken (seconds)')
    plt.legend()
    plt.savefig('figures/knn_spotify/time_vs_train_size.png')
    plt.close()

    #plot a bar graph for the time taken for the different models at 90000 train size
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    times = [results[model][-1] for model in models]
    plt.bar(models, times)
    plt.title('Time Taken for Different Models on Train Set')
    plt.ylabel('Time Taken (seconds)')
    plt.xlabel('Model')
    plt.savefig('figures/knn_spotify/knn_time_bar_graph.png')
    plt.close()

def knn_time_analysis():
    file_path = 'results/knn_results_time.txt'
    train_sizes, results = parse_knn_time_results(file_path)
    plot_time_vs_train_size(train_sizes, results)

def drop_random_columns():
    train = pd.read_csv('../../data/interim/spotify_v1/train.csv')
    test = pd.read_csv('../../data/interim/spotify_v1/test.csv')

    X_train = train.drop(columns=['track_genre']).values
    y_train = train['track_genre'].values

    X_test = test.drop(columns=['track_genre']).values
    y_test = test['track_genre'].values

    num_columns = X_train.shape[1]

    with open("results/knn_drop_columns.txt", "w") as f:
        f.write("Dropping Random Columns\n")

    for i in range(5, num_columns+1):
        print("Number of columns", i)
        columns = list(range(num_columns))
        import random
        random.shuffle(columns)
        columns = columns[:i]
        column_names = train.columns[columns]
        X_train_subset = X_train[:, columns]
        X_test_subset = X_test[:, columns]

        knn = OptimisedKNN(20, 'manhattan')
        knn.fit(X_train_subset, y_train)
        y_pred = knn.predict(X_test_subset)

        accuracy = Performance_Measures().accuracy(y_test, y_pred)

        with open("results/knn_drop_columns.txt", "a") as f:
            f.write(f"Number of columns: {i}\n")
            f.write(f"Accuracy: {accuracy}\n")
            for column in column_names:
                f.write(f"{column}, ")
            f.write("\n")

def plot_columns_vs_accuracy():

    file_path = 'results/knn_drop_columns.txt'
    num_columns = []
    accuracies = []
    columns_used = []

    with open(file_path, 'r') as file:
        for line in file:
            num_cols_match = re.search(r'Number of columns: (\d+)', line)
            if num_cols_match:
                num_columns.append(int(num_cols_match.group(1)))

            accuracy_match = re.search(r'Accuracy: ([\d\.]+)', line)
            if accuracy_match:
                accuracies.append(float(accuracy_match.group(1)))

            if ',' in line:
                columns = line.strip().split(', ')
                if columns[-1] == '':
                    columns.pop() 
                    # to remove the last empty string after ,
                columns_used.append(', '.join(columns))

    colors = plt.cm.viridis(np.linspace(0, 1, len(num_columns)))

    plt.figure(figsize=(12, 6))
    for i in range(len(num_columns)):
        plt.bar(num_columns[i], accuracies[i], color=colors[i], label=f"{num_columns[i]} cols: {columns_used[i]}")

    plt.title('Number of Columns vs Accuracy')
    plt.xlabel('Number of Columns')
    plt.ylabel('Accuracy')
    plt.savefig('figures/knn_spotify/columns_vs_accuracy.png')


    plt.figure(figsize=(12, 6))
    for i in range(len(num_columns)):
        plt.plot([], [], marker='o', color=colors[i], linestyle='-', label=f"{num_columns[i]} cols: {columns_used[i]}")

    plt.legend(loc='center', fontsize='small')
    plt.axis('off')  
    plt.savefig('figures/knn_spotify/columns_legend.png')

def second_dataset():
    train = pd.read_csv('../../data/interim/spotify_2/train.csv')
    test = pd.read_csv('../../data/interim/spotify_2/test.csv')

    X_train = train.drop(columns=['track_genre']).values
    y_train = train['track_genre'].values

    X_test = test.drop(columns=['track_genre']).values
    y_test = test['track_genre'].values

    knn = OptimisedKNN(20, 'manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = Performance_Measures().accuracy(y_test, y_pred)
    print("Accuracy: ", accuracy)
    
    precision = Performance_Measures().precision(y_test, y_pred)
    print("Precision: ", precision)

    recall = Performance_Measures().recall(y_test, y_pred)
    print("Recall: ", recall)

    f1 = Performance_Measures().f1_score(y_test, y_pred)
    print("F1 Score: ", f1)

def best_5_columns():
    columns = ['danceability', 'valence', 'liveness', 'popularity', 'acousticness']
    train = pd.read_csv('../../data/interim/spotify_2/train.csv')
    test = pd.read_csv('../../data/interim/spotify_2/test.csv')

    train = train[columns + ['track_genre']]
    test = test[columns + ['track_genre']]

    X_train = train.drop(columns=['track_genre']).values
    y_train = train['track_genre'].values

    X_test = test.drop(columns=['track_genre']).values
    y_test = test['track_genre'].values

    knn = OptimisedKNN(20, 'manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = Performance_Measures().accuracy(y_test, y_pred)
    print("Accuracy: ", accuracy)
    
    precision = Performance_Measures().precision(y_test, y_pred)
    print("Precision: ", precision)

    recall = Performance_Measures().recall(y_test, y_pred)
    print("Recall: ", recall)

    f1 = Performance_Measures().f1_score(y_test, y_pred)
    print("F1 Score: ", f1)

# hyperparameter_tuning()
# time_for_pred_based_on_train_size()
# best_k_distance_metric()
# knn_time_analysis()
# drop_random_columns()
# plot_columns_vs_accuracy()
# second_dataset()
# best_5_columns()

