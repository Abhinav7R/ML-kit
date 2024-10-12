"""
This script is to implement all tasks in assignment-3 questions in task 4.
AutoEncoders
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.AutoEncoders.auto_encoders import AutoEncoder
from models.knn.knn import OptimisedKNN
from performance_measures.performance_measures import Performance_Measures
from models.MLP.mlp import MLP

import numpy as np
import pandas as pd

def performance_metrics(y, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    performance = Performance_Measures()
    accuracy = performance.accuracy(y, y_pred)
    precision = performance.precision(y, y_pred)
    recall = performance.recall(y, y_pred)
    f1_score = performance.f1_score(y, y_pred)
    return accuracy, precision, recall, f1_score


def test_auto_encoder_on_spotify():
    train_data = "../../data/interim/spotify_v1/train.csv"
    df = pd.read_csv(train_data)
    X = df.iloc[:, :-1].values

    val_data = "../../data/interim/spotify_v1/val.csv"
    df = pd.read_csv(val_data)
    X_val = df.iloc[:, :-1].values

    auto_encoder = AutoEncoder(l_r=0.01,
                                 activation_function='sigmoid',
                                 optimizer='mini_batch_gradient_descent',
                                 hidden_layers=1,
                                 neurons_per_layer=[10],
                                 reduced_dimension=6,
                                 batch_size=32,
                                 no_of_epochs=200,
                                 input_layer_size=X.shape[1])
    
    auto_encoder.fit(X, val=True, X_val=X_val, early_stopping=True, patience=5, wandb_log=False)
    print("Reconstruction Loss: ", auto_encoder.reconstruction_loss(X_val))

def get_spotify_reduced():
    data = "../../data/processed/spotify/spotify_v1.csv"
    df = pd.read_csv(data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    auto_encoder = AutoEncoder(l_r=0.01,
                                 activation_function='sigmoid',
                                 optimizer='mini_batch_gradient_descent',
                                 hidden_layers=1,
                                 neurons_per_layer=[10],
                                 reduced_dimension=6,
                                 batch_size=32,
                                 no_of_epochs=200,
                                 input_layer_size=X.shape[1])
    
    auto_encoder.fit(X)
    reconstruction_loss = auto_encoder.reconstruction_loss(X)
    print("Reconstruction Loss: ", reconstruction_loss)

    latent_train = auto_encoder.get_latent(X)
    

    # path to save the latent representation
    latent_path = "../../data/processed/spotify/spotify_from_auto_encoder.csv"
    latent_df = pd.DataFrame(latent_train)
    latent_df['target_genre'] = y
    latent_df.to_csv(latent_path, index=False)


def knn_on_auto_encoded_spotify():
    latent_path = "../../data/processed/spotify/spotify_from_auto_encoder.csv"
    df = pd.read_csv(latent_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # split data
    test_split = 0.2

    X, X_test = np.split(X, [int((1-test_split)*X.shape[0])])
    y, y_test = np.split(y, [int((1-test_split)*y.shape[0])])

    knn = OptimisedKNN(k=20, distance_metric='manhattan')
    knn.fit(X, y)
    y_pred = knn.predict(X_test)
    accuracy = Performance_Measures().accuracy(y_test, y_pred)
    print("Accuracy: ", accuracy)
    
    precision = Performance_Measures().precision(y_test, y_pred)
    print("Precision: ", precision)

    recall = Performance_Measures().recall(y_test, y_pred)
    print("Recall: ", recall)

    f1 = Performance_Measures().f1_score(y_test, y_pred)
    print("F1 Score: ", f1)


def mlp_on_spotify():
    data = "../../data/processed/spotify/spotify_from_auto_encoder.csv"
    df = pd.read_csv(data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # encode the y values
    unique_labels = np.unique(y)
    y_encoded = np.zeros((y.shape[0], len(unique_labels)))
    encoding = {unique_labels[i]: i for i in range(len(unique_labels))}
    for i in range(y.shape[0]):
        y_encoded[i, encoding[y[i]]] = 1
    y = y_encoded

    test_split = 0.2

    X, X_test = np.split(X, [int((1-test_split)*X.shape[0])])
    y, y_test = np.split(y, [int((1-test_split)*y.shape[0])])

    mlp = MLP(l_r=0.01,
              activation_function='sigmoid',
              optimizer='mini_batch_gradient_descent',
              hidden_layers=3,
              neurons_per_layer=[25, 75, 120],
              batch_size=32,
              no_of_epochs=200,
              input_layer_size=X.shape[1],
              output_layer_size=y.shape[1],
              task='classification')
    
    mlp.fit(X, y)
    y_pred = mlp.predict(X_test)
    a, p, r, f1 = performance_metrics(y_test, y_pred)
    print("Accuracy: ", a)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1 Score: ", f1)


# test_auto_encoder_on_spotify()
# get_spotify_reduced()
# knn_on_auto_encoded_spotify()
mlp_on_spotify()
