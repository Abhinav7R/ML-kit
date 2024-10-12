"""
This script is to implement all tasks in assignment 3.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.MLP.mlp import MLP
from performance_measures.performance_measures import Performance_Measures

import numpy as np
import pandas as pd
import json

def performance_metrics(y, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    performance = Performance_Measures()
    accuracy = performance.accuracy(y, y_pred)
    precision = performance.precision(y, y_pred)
    recall = performance.recall(y, y_pred)
    f1_score = performance.f1_score(y, y_pred)
    return accuracy, precision, recall, f1_score

def split_data_classification(only_test=False):
    data_path = "../../data/processed/wineQT/wine_normalised.csv"
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(-1, 1)
    # one hot encoding for y
    y_unique = np.unique(y)
    y_encoded = np.zeros((len(y), len(y_unique)))
    encoding = {y_unique[i]: i for i in range(len(y_unique))}
    for i in range(len(y)):
        y_encoded[i, encoding[y[i][0]]] = 1
    y = y_encoded

    test_split = 0.2
    val_split = 0.2
    X, X_test = np.split(X, [int((1-test_split)*X.shape[0])])
    y, y_test = np.split(y, [int((1-test_split)*y.shape[0])])
    if only_test:
        return X, y, X_test, y_test
    X, X_val = np.split(X, [int((1-val_split)*X.shape[0])])
    y, y_val = np.split(y, [int((1-val_split)*y.shape[0])])

    return X, y, X_val, y_val, X_test, y_test

def testing_mlp():
    X, y, X_val, y_val, X_test, y_test = split_data_classification()

    mlp = MLP(l_r=0.01,
              activation_function='relu',
              optimizer='mini_batch_gradient_descent',
              hidden_layers=2,
              neurons_per_layer=[16, 10],
              batch_size=32,
              no_of_epochs=100,
              input_layer_size=X.shape[1],
              output_layer_size=y.shape[1],
              task='classification')

    mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)

    y_pred = mlp.predict(X_val)
    val_accuracy, val_precision, val_recall, val_f1_score = performance_metrics(y_val, y_pred)
    y_train_pred = mlp.predict(X)
    train_accuracy, train_precision, train_recall, train_f1_score = performance_metrics(y, y_train_pred)
    print("Validation Accuracy: ", val_accuracy)
    print("Validation Precision: ", val_precision)
    print("Validation Recall: ", val_recall)
    print("Validation F1 Score: ", val_f1_score)
    print("Train Accuracy: ", train_accuracy)
    print("Train Precision: ", train_precision)
    print("Train Recall: ", train_recall)
    print("Train F1 Score: ", train_f1_score)


    y_pred = mlp.predict(X_test)
    test_accuracy, test_precision, test_recall, test_f1_score = performance_metrics(y_test, y_pred)
    print("Test Accuracy: ", test_accuracy)
    print("Test Precision: ", test_precision)
    print("Test Recall: ", test_recall)
    print("Test F1 Score: ", test_f1_score)
    

def performance_metrics_multilabel(y, y_pred):
    threshold = 0.5
    y_pred = np.where(y_pred > threshold, 1, 0)
    y = np.where(y > threshold, 1, 0)
    # accuracy = accuracy_score(y, y_pred)
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == y[i][j] and y[i][j] == 1:
                tp += 1
            elif y_pred[i][j] == y[i][j] and y[i][j] == 0:
                tn += 1
            elif y_pred[i][j] != y[i][j] and y[i][j] == 1:
                fn += 1
            elif y_pred[i][j] != y[i][j] and y[i][j] == 0:
                fp += 1
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    hamming_loss = np.mean(y != y_pred)

    return accuracy, precision, recall, f1, hamming_loss
   

def split_data_multilabel(only_test=False):
    data_path = "../../data/processed/advertisement/advertisement.csv"
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # multi label encoding
    y_unique = []
    for i in range(len(y)):
        label = y[i].split(' ')
        for l in label:
            if l not in y_unique:
                y_unique.append(l)
    # print(y_unique)
    y_encoded = np.zeros((len(y), len(y_unique)))
    encoding = {y_unique[i]: i for i in range(len(y_unique))}
    for i in range(len(y)):
        label = y[i].split(' ')
        for l in label:
            y_encoded[i, encoding[l]] = 1
    y = y_encoded
    # print(y)

    # save the encoding
    with open("figures/advertisement_encoding.json", "w") as file:
        json.dump(encoding, file, indent=4)

    test_split = 0.2
    val_split = 0.2
    X, X_test = np.split(X, [int((1-test_split)*X.shape[0])])
    y, y_test = np.split(y, [int((1-test_split)*y.shape[0])])
    if only_test:
        return X, y, X_test, y_test
    X, X_val = np.split(X, [int((1-val_split)*X.shape[0])])
    y, y_val = np.split(y, [int((1-val_split)*y.shape[0])])

    return X, y, X_val, y_val, X_test, y_test

def testing_mlp_multilabel():
    X, y, X_val, y_val, X_test, y_test = split_data_multilabel()

    mlp = MLP(l_r=0.01,
              activation_function='sigmoid',
              optimizer='batch_gradient_descent',
              hidden_layers=2,
              neurons_per_layer=[16, 10],
              batch_size=32,
              no_of_epochs=100,
              input_layer_size=X.shape[1],
              output_layer_size=y.shape[1],
              task='multilabel_classification')

    mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)

    y_pred = mlp.predict(X_val)
    val_accuracy, val_precision, val_recall, val_f1_score, val_hamming_loss = performance_metrics_multilabel(y_val, y_pred)
    y_train_pred = mlp.predict(X)
    train_accuracy, train_precision, train_recall, train_f1_score, hamming_loss_train = performance_metrics_multilabel(y, y_train_pred)
    print("Validation Accuracy: ", val_accuracy)
    print("Validation Precision: ", val_precision)
    print("Validation Recall: ", val_recall)
    print("Validation F1 Score: ", val_f1_score)
    print("Validation Hamming Loss: ", val_hamming_loss)
    print()
    print("Train Accuracy: ", train_accuracy)
    print("Train Precision: ", train_precision)
    print("Train Recall: ", train_recall)
    print("Train F1 Score: ", train_f1_score)
    print("Train Hamming Loss: ", hamming_loss_train)

    y_pred = mlp.predict(X_test)
    test_accuracy, test_precision, test_recall, test_f1_score, test_hamming_loss = performance_metrics_multilabel(y_test, y_pred)
    print("Test Accuracy: ", test_accuracy)
    print("Test Precision: ", test_precision)
    print("Test Recall: ", test_recall)
    print("Test F1 Score: ", test_f1_score)
    print("Test Hamming Loss: ", test_hamming_loss)

def performance_metrics_regression(y, y_pred):

    #mse
    mse = np.mean((y - y_pred) ** 2)
    #mae
    mae = np.mean(np.abs(y - y_pred))
    #rmse
    rmse = np.sqrt(mse)
    #r2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, mae, rmse, r2
   

def split_data(only_test=False):
    data_path = "../../data/processed/HousingData/HousingData_normalised.csv"
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(-1, 1)

    test_split = 0.2
    val_split = 0.2
    X, X_test = np.split(X, [int((1-test_split)*X.shape[0])])
    y, y_test = np.split(y, [int((1-test_split)*y.shape[0])])
    if only_test:
        return X, y, X_test, y_test
    X, X_val = np.split(X, [int((1-val_split)*X.shape[0])])
    y, y_val = np.split(y, [int((1-val_split)*y.shape[0])])

    return X, y, X_val, y_val, X_test, y_test

def testing_mlp_regression():
    X, y, X_val, y_val, X_test, y_test = split_data()

    mlp = MLP(l_r=0.001,
              activation_function='tanh',
              optimizer='sgd',
              hidden_layers=3,
              neurons_per_layer=[16, 10, 5],
              batch_size=64,
              no_of_epochs=400,
              input_layer_size=X.shape[1],
              output_layer_size=1,
              task='regression')

    mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)

    y_pred = mlp.predict(X_val)
    mse, mae, rmse, r2 = performance_metrics_regression(y_val, y_pred)
    print("Validation")
    print(f"mse: {mse}, mae: {mae}, rmse: {rmse}, r2: {r2}")

    y_train_pred = mlp.predict(X)
    mse, mae, rmse, r2 = performance_metrics_regression(y, y_train_pred)
    print("Train")
    print(f"mse: {mse}, mae: {mae}, rmse: {rmse}, r2: {r2}")

    y_pred = mlp.predict(X_test)
    mse, mae, rmse, r2 = performance_metrics_regression(y_test, y_pred)
    print("Test")
    print(f"mse: {mse}, mae: {mae}, rmse: {rmse}, r2: {r2}")



# testing_mlp()
# testing_mlp_multilabel()
# testing_mlp_regression()

