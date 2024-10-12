"""
This script is to implement all tasks in assignment-3 questions in task 2.
Multilayer Perceptron (MLP) for Classification with Multilabel
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.MLP.mlp_classification_multilabel import MLP_Multilabel_Classification as MLP

import numpy as np
import pandas as pd
import wandb
import prettytable as pt
import json
import matplotlib.pyplot as plt

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
   

def split_data(only_test=False):
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
    X, y, X_val, y_val, X_test, y_test = split_data()

    mlp = MLP(l_r=0.01,
              activation_function='sigmoid',
              optimizer='sgd',
              hidden_layers=2,
              neurons_per_layer=[16, 10],
              batch_size=32,
              no_of_epochs=100,
              input_layer_size=X.shape[1],
              output_layer_size=y.shape[1])

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

def sweep_train():
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data()

    with wandb.init() as run:
        config = run.config
        wandb.run.name = f"l_r_{config.l_r}_act_{config.activation_function}_opt_{config.optimizer}_hl_{len(config.neurons_per_layer)}_neurons_{config.neurons_per_layer}_epochs_{config.no_of_epochs}"
        model = MLP(
            l_r=config.l_r,
            activation_function=config.activation_function,
            optimizer=config.optimizer,
            hidden_layers=len(config.neurons_per_layer),
            neurons_per_layer=config.neurons_per_layer,
            no_of_epochs=config.no_of_epochs,
            input_layer_size=X_train.shape[1],
            output_layer_size=y_train.shape[1]
        )
        
        model.fit(X_train, y_train, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=True)
        y_pred = model.predict(X_val)
        val_accuracy, val_precision, val_recall, val_f1_score, val_hamming_loss = performance_metrics_multilabel(y_val, y_pred)
        y_train_pred = model.predict(X_train)
        train_accuracy, train_precision, train_recall, train_f1_score, train_hamming_loss = performance_metrics_multilabel(y_train, y_train_pred)
        wandb.log({"val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_f1_score": val_f1_score, "val_hamming_loss": val_hamming_loss,
                   "train_accuracy": train_accuracy, "train_precision": train_precision, "train_recall": train_recall, "train_f1_score": train_f1_score, "train_hamming_loss": train_hamming_loss})
        
def hyperparameter_tuning():
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'l_r': {
                'values': [0.01, 0.001]
            },
            'activation_function': {
                'values': ['relu', 'sigmoid', 'tanh']
            },
            'optimizer': {
                'values': ['mini_batch_gradient_descent', 'batch_gradient_descent', 'sgd']
            },
            'neurons_per_layer': {
                'values': [[8], [16, 10], [16, 10, 5]]
            },
            'no_of_epochs': {
                'values': [100, 200]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="SMAI_ASG_3_Multilabel_Classification")
    wandb.agent(sweep_id, function=sweep_train)

def generate_table():

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("abhinav7/SMAI_ASG_3_Multilabel_Classification")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

    runs_df.to_csv("runs/MLP_classification_multilabel.csv")
                   
    # generate a table using prettytable
    x = pt.PrettyTable()
    x.field_names = ["lr", "activation_function", "optimizer", "neurons_per_layer", "epochs", "val_accuracy", 
                     "val_precision", "val_recall", "val_f1_score", "train_accuracy", "train_precision", "train_recall", "train_f1_score",
                     "train_hamming_loss", "val_hamming_loss"]

    for i in range(len(runs_df)):
        x.add_row([runs_df.config[i]["l_r"], runs_df.config[i]["activation_function"], runs_df.config[i]["optimizer"], 
                   runs_df.config[i]["neurons_per_layer"], runs_df.summary[i]["epoch"], runs_df.summary[i]["val_accuracy"], 
                   runs_df.summary[i]["val_precision"], runs_df.summary[i]["val_recall"], runs_df.summary[i]["val_f1_score"], 
                   runs_df.summary[i]["train_accuracy"], runs_df.summary[i]["train_precision"], runs_df.summary[i]["train_recall"], 
                   runs_df.summary[i]["train_f1_score"], runs_df.summary[i]["train_hamming_loss"], runs_df.summary[i]["val_hamming_loss"]])
    x.set_style(pt.MARKDOWN)
    
    # write the table to a file
    with open("runs/MLP_classification_multilabel_table.txt", "w") as file:
        file.write(str(x))

def get_best_model():
    df = pd.read_csv("runs/MLP_classification_multilabel.csv")
    best_run = df.loc[df["summary"].apply(lambda x: eval(x)["val_accuracy"]).idxmax()]
    best_config = eval(best_run["config"])
    
    print("Best Hyper parameters: ")
    for key, value in best_config.items():
        print(key, ": ", value)

    with open("runs/MLP_classification_multilabel_best_hyperparams.json", "w") as file:
        json.dump(best_config, file, indent=4)


def get_performance_of_best_model():
    with open("runs/MLP_classification_multilabel_best_hyperparams.json", "r") as file:
        best_config = json.load(file)

    X, y, X_test, y_test = split_data(only_test=True)

    model = MLP(
        l_r=best_config["l_r"],
        activation_function=best_config["activation_function"],
        optimizer=best_config["optimizer"],
        hidden_layers=len(best_config["neurons_per_layer"]),
        neurons_per_layer=best_config["neurons_per_layer"],
        no_of_epochs=best_config["no_of_epochs"],
        input_layer_size=X.shape[1],
        output_layer_size=y.shape[1]
    )

    model.fit(X, y, early_stopping=True, patience=5, wandb_log=False)
    y_pred = model.predict(X_test)
    
    accuracy, precision, recall, f1, hamming_loss = performance_metrics_multilabel(y_test, y_pred)
    print("Test Accuracy: ", accuracy)
    print("Test Precision: ", precision)
    print("Test Recall: ", recall)
    print("Test F1 Score: ", f1)
    print("Test Hamming Loss: ", hamming_loss)
    
    model.save_model("runs/MLP_classification_multilabel_best_model.pkl")


def analysis_of_classification():
    with open("runs/MLP_classification_multilabel_best_hyperparams.json", "r") as file:
        best_config = json.load(file)

    X, y, X_test, y_test = split_data(only_test=True)

    model = MLP(
        l_r=best_config["l_r"],
        activation_function=best_config["activation_function"],
        optimizer=best_config["optimizer"],
        hidden_layers=len(best_config["neurons_per_layer"]),
        neurons_per_layer=best_config["neurons_per_layer"],
        no_of_epochs=best_config["no_of_epochs"],
        input_layer_size=X.shape[1],
        output_layer_size=y.shape[1]
    )

    model.fit(X, y, early_stopping=True, patience=5, wandb_log=False)
    y_pred = model.predict(X_test)
    
    # for each class calculate accuracy, precision, recall, f1 score
    y_pred = np.where(y_pred > 0.4, 1, 0)
    y_test = np.where(y_test > 0.4, 1, 0)
    stats = []

    for i in range(y_test.shape[1]):
        a, p, r, f, h = performance_metrics_multilabel(y_test[:, i].reshape(-1, 1), y_pred[:, i].reshape(-1, 1))
        stats.append([i, a, p, r, f, h])    
    stats_df = pd.DataFrame(stats, columns=["Class", "Accuracy", "Precision", "Recall", "F1 Score", "Hamming Loss"])
    
    encoding = {}
    with open("figures/advertisement_encoding.json", "r") as file:
        encoding = json.load(file)

    stats_df["Class"] = stats_df["Class"].apply(lambda x: list(encoding.keys())[x])
    
    # print(stats_df)
    table = pt.PrettyTable()
    table.field_names = stats_df.columns
    for i in range(len(stats_df)):
        table.add_row(stats_df.iloc[i])
    table.set_style(pt.MARKDOWN)
    print(table)

def check_class_balancing():
    data_path = "../../data/processed/advertisement/advertisement.csv"
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1].values
    y_unique = []
    for i in range(len(y)):
        label = y[i].split(' ')
        for l in label:
            if l not in y_unique:
                y_unique.append(l)
    y_encoded = np.zeros((len(y), len(y_unique)))
    encoding = {y_unique[i]: i for i in range(len(y_unique))}
    for i in range(len(y)):
        label = y[i].split(' ')
        for l in label:
            y_encoded[i, encoding[l]] = 1
    y = y_encoded
    class_count = np.sum(y, axis=0)
    print("Class Count: ", class_count)
    plt.bar(range(len(class_count)), class_count)
    plt.xticks(range(len(class_count)), list(encoding.keys()))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.savefig("figures/advertisement_class_distribution.png")
    plt.show()


# testing_mlp_multilabel()
# hyperparameter_tuning()
# generate_table()
# get_best_model()
# get_performance_of_best_model()
# analysis_of_classification()
# check_class_balancing()





