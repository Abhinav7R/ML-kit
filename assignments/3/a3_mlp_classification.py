"""
This script is to implement all tasks in assignment-3 questions in task 2.
Multilayer Perceptron (MLP) for Classification
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.MLP.mlp_classification import MLP_Classification as MLP
from performance_measures.performance_measures import Performance_Measures

import numpy as np
import pandas as pd
import wandb
import prettytable as pt
import json
import matplotlib.pyplot as plt


def performance_metrics(y, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    performance = Performance_Measures()
    accuracy = performance.accuracy(y, y_pred)
    precision = performance.precision(y, y_pred)
    recall = performance.recall(y, y_pred)
    f1_score = performance.f1_score(y, y_pred)
    return accuracy, precision, recall, f1_score

def split_data(only_test=False):
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
    X, y, X_val, y_val, X_test, y_test = split_data()

    mlp = MLP(l_r=0.01,
              activation_function='relu',
              optimizer='mini_batch_gradient_descent',
              hidden_layers=2,
              neurons_per_layer=[16, 10],
              batch_size=32,
              no_of_epochs=100,
              input_layer_size=X.shape[1],
              output_layer_size=y.shape[1])

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
        val_accuracy, val_precision, val_recall, val_f1_score = performance_metrics(y_val, y_pred)
        y_train_pred = model.predict(X_train)
        train_accuracy, train_precision, train_recall, train_f1_score = performance_metrics(y_train, y_train_pred)
        wandb.log({"val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_f1_score": val_f1_score,
                   "train_accuracy": train_accuracy, "train_precision": train_precision, "train_recall": train_recall, "train_f1_score": train_f1_score})
        
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

    sweep_id = wandb.sweep(sweep_config, project="SMAI_ASG_3_Single_Label_Classification")
    wandb.agent(sweep_id, function=sweep_train)

def generate_table():

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("abhinav7/SMAI_ASG_3_Single_Label_Classification")

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

    runs_df.to_csv("runs/MLP_classifcation.csv")

    # generate a table using prettytable
    x = pt.PrettyTable()
    x.field_names = ["lr", "activation_function", "optimizer", "neurons_per_layer", "epochs", "val_accuracy", 
                     "val_precision", "val_recall", "val_f1_score", "train_accuracy", "train_precision", "train_recall", "train_f1_score"]

    for i in range(len(runs_df)):
        x.add_row([runs_df.config[i]["l_r"], runs_df.config[i]["activation_function"], runs_df.config[i]["optimizer"], 
                   runs_df.config[i]["neurons_per_layer"], runs_df.summary[i]["epoch"], runs_df.summary[i]["val_accuracy"], 
                   runs_df.summary[i]["val_precision"], runs_df.summary[i]["val_recall"], runs_df.summary[i]["val_f1_score"], 
                   runs_df.summary[i]["train_accuracy"], runs_df.summary[i]["train_precision"], runs_df.summary[i]["train_recall"], 
                   runs_df.summary[i]["train_f1_score"]])
    x.set_style(pt.MARKDOWN)
    
    # write the table to a file
    with open("runs/MLP_classification_table.txt", "w") as file:
        file.write(str(x))

def get_best_model():
    
    # read the csv file to get the best model hyper params (lowest val_loss)
    df = pd.read_csv("runs/MLP_classifcation.csv")
    
    best_run = df.loc[df["summary"].apply(lambda x: eval(x)["val_accuracy"]).idxmax()]
    best_config = eval(best_run["config"])
    
    print("Best Hyper parameters: ")
    for key, value in best_config.items():
        print(key, ": ", value)

    with open("runs/MLP_classification_best_hyperparams.json", "w") as file:
        json.dump(best_config, file, indent=4)

def get_performance_of_best_model():
    with open("runs/MLP_classification_best_hyperparams.json", "r") as file:
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
    test_accuracy, test_precision, test_recall, test_f1_score = performance_metrics(y_test, y_pred)
    print("Test Accuracy: ", test_accuracy)
    print("Test Precision: ", test_precision)
    print("Test Recall: ", test_recall)
    print("Test F1 Score: ", test_f1_score)

    model.save_model("runs/MLP_classification_best_model.pkl")

def analysing_hyperparams_activation():
    with open("runs/MLP_classification_best_hyperparams.json", "r") as file:
        best_config = json.load(file)

    activation_functions = ["relu", "sigmoid", "tanh", "linear"]

    X, y, X_val, y_val, X_test, y_test = split_data()

    losses_dict = {}

    for activation in activation_functions:
        print(f"Training model with activation function: {activation}")
        
        model = MLP(
            l_r=best_config["l_r"],
            activation_function=activation,
            optimizer=best_config["optimizer"],
            hidden_layers=len(best_config["neurons_per_layer"]),
            neurons_per_layer=best_config["neurons_per_layer"],
            no_of_epochs=best_config["no_of_epochs"],
            input_layer_size=X.shape[1],
            output_layer_size=y.shape[1]
        )

        losses = model.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
        losses_dict[activation] = losses

    # print(losses_dict)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for key, value in losses_dict.items():
        ax.plot(value, label=key)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss(val)")
    ax.set_title("Loss vs Epochs for different activation functions")
    ax.legend()
    plt.savefig("figures/MLP_classification_activation_functions.png")

    print("Plot saved to figures/MLP_classification_activation_functions.png")

def analysing_hyperparams_lr():
    with open("runs/MLP_classification_best_hyperparams.json", "r") as file:
        best_config = json.load(file)

    l_r_values = [0.1, 0.01, 0.001, 0.0001]

    X, y, X_val, y_val, X_test, y_test = split_data()

    losses_dict = {}

    for l_r in l_r_values:
        print(f"Training model with learning rate: {l_r}")

        model = MLP(
            l_r=l_r,
            activation_function=best_config["activation_function"],
            optimizer=best_config["optimizer"],
            hidden_layers=len(best_config["neurons_per_layer"]),
            neurons_per_layer=best_config["neurons_per_layer"],
            no_of_epochs=best_config["no_of_epochs"],
            input_layer_size=X.shape[1],
            output_layer_size=y.shape[1]
        )

        losses = model.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
        losses_dict[l_r] = losses

    # print(losses_dict)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for key, value in losses_dict.items():
        ax.plot(value, label=key)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss(val)")
    ax.set_title("Loss vs Epochs for different learning rates")
    ax.legend()
    plt.savefig("figures/MLP_classification_learning_rates.png")

    print("Plot saved to figures/MLP_classification_learning_rates.png")

def analysing_hyperparams_batch_size():
    with open("runs/MLP_classification_best_hyperparams.json", "r") as file:
        best_config = json.load(file)

    batch_size = [8, 64, 128, 256]

    X, y, X_val, y_val, X_test, y_test = split_data()

    losses_dict = {}

    for b_s in batch_size:
        print(f"Training model with batch size: {b_s}")

        model = MLP(
            l_r=best_config["l_r"],
            activation_function=best_config["activation_function"],
            optimizer="mini_batch_gradient_descent",
            hidden_layers=len(best_config["neurons_per_layer"]),
            neurons_per_layer=best_config["neurons_per_layer"],
            no_of_epochs=best_config["no_of_epochs"],
            input_layer_size=X.shape[1],
            output_layer_size=y.shape[1],
            batch_size=b_s
        )

        losses = model.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
        losses_dict[b_s] = losses

    # print(losses_dict)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for key, value in losses_dict.items():
        ax.plot(value, label=key)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss(val)")
    ax.set_title("Loss vs Epochs for different batch sizes")
    ax.legend()
    plt.savefig("figures/MLP_classification_batch_sizes.png")

    print("Plot saved to figures/MLP_classification_batch_sizes.png")

def analysis_of_classification():
    with open("runs/MLP_classification_best_hyperparams.json", "r") as file:
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
    # print(y_test.shape, y_pred.shape)
    # print(y_test[:5], y_pred[:5])
    # for every class calculate the accuracy, precision, recall, f1_score
    stats = []
    for i in range(y_test.shape[1]):
        
        y_test_class = y_test[:, i]
        y_pred_class = y_pred[:, i]
        
        y_pred_class = (y_pred_class >= 0.5).astype(int)


        tp = np.sum(y_test_class * y_pred_class)
        tn = np.sum((1 - y_test_class) * (1 - y_pred_class))
        fp = np.sum((1 - y_test_class) * y_pred_class)
        fn = np.sum(y_test_class * (1 - y_pred_class))
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats.append([accuracy, precision, recall, f1_score])
        print(tp, tn, fp, fn)

    stats_df = pd.DataFrame(stats, columns=["accuracy", "precision", "recall", "f1_score"])

    table = pt.PrettyTable()
    table.field_names = stats_df.columns
    for i in range(len(stats_df)):
        table.add_row(stats_df.iloc[i])
    table.set_style(pt.MARKDOWN)
    print(table)

def plot_wine_distribution():
    data_path = "../../data/processed/wineQT/wine_normalised.csv"
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1].values
    y = y*5+3
    # violin plot
    plt.figure(figsize=(6, 6))
    plt.violinplot(y)
    plt.xlabel("Class")
    plt.ylabel("Value")
    plt.title("Distribution of classes")
    plt.savefig("figures/wine_quality_distribution.png")

testing_mlp()
# hyperparameter_tuning()
# generate_table()
# get_best_model()
# get_performance_of_best_model()
# analysing_hyperparams_activation()
# analysing_hyperparams_lr()
# analysing_hyperparams_batch_size()
# analysis_of_classification()
# plot_wine_distribution()

