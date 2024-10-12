"""
This script is to implement all tasks in assignment-3 questions in task 3.
Multilayer Perceptron (MLP) for Regression
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

from models.MLP.mlp_regression import MLP_Regression as MLP

import numpy as np
import pandas as pd
import wandb
import prettytable as pt
import json
import matplotlib.pyplot as plt
import seaborn as sns

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
              output_layer_size=1)
    mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
    y_pred = mlp.predict(X_val)
    mse, mae, rmse, r2 = performance_metrics_regression(y_val, y_pred)
    print(f"mse: {mse}, mae: {mae}, rmse: {rmse}, r2: {r2}")

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
            output_layer_size=1
        )
        
        model.fit(X_train, y_train, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=True)
        y_pred = model.predict(X_val)
        mse, mae, rmse, r2 = performance_metrics_regression(y_val, y_pred)
        wandb.log({"mse_val": mse, "mae_val": mae, "rmse_val": rmse, "r2_val": r2})
        y_pred = model.predict(X_train)
        mse, mae, rmse, r2 = performance_metrics_regression(y_train, y_pred)
        wandb.log({"mse_train": mse, "mae_train": mae, "rmse_train": rmse, "r2_train": r2})
        

def hyperparameter_tuning():
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'l_r': {
                'values': [0.001, 0.0001]
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
                'values': [200, 400]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="SMAI_ASG_3_MLP_Regression")
    wandb.agent(sweep_id, function=sweep_train)


def generate_table():
        
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("abhinav7/SMAI_ASG_3_MLP_Regression")

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

    runs_df.to_csv("runs/MLP_Regression.csv")

    x = pt.PrettyTable()
    x.field_names = ["l_r", "activation_function", "optimizer", "neurons_per_layer", "epochs", "mse_val", "mae_val", "rmse_val", "r2_val", "mse_train", "mae_train", "rmse_train", "r2_train"]

    for i in range(len(runs_df)):
        x.add_row([runs_df.config[i]['l_r'], runs_df.config[i]['activation_function'], runs_df.config[i]['optimizer'], runs_df.config[i]['neurons_per_layer'], runs_df.summary[i]['epoch'], runs_df.summary[i]['mse_val'], runs_df.summary[i]['mae_val'], runs_df.summary[i]['rmse_val'], runs_df.summary[i]['r2_val'], runs_df.summary[i]['mse_train'], runs_df.summary[i]['mae_train'], runs_df.summary[i]['rmse_train'], runs_df.summary[i]['r2_train']])
    x.set_style(pt.MARKDOWN)

    with open("runs/MLP_Regression_table.txt", "w") as f:
        f.write(str(x))

def get_best_model():
    df = pd.read_csv("runs/MLP_Regression.csv")
    best_run = df.loc[df["summary"].apply(lambda x: eval(x)["mse_val"]).idxmin()]
    best_config = eval(best_run["config"])
    
    print("Best Hyper parameters: ")
    for key, value in best_config.items():
        print(key, ": ", value)

    with open("runs/MLP_Regression_best_hyperparams.json", "w") as file:
        json.dump(best_config, file, indent=4)

def get_performance_of_best_model():
    with open("runs/MLP_Regression_best_hyperparams.json", "r") as file:
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
        output_layer_size=1
    )

    model.fit(X, y, early_stopping=True, patience=5, wandb_log=False)
    y_pred = model.predict(X_test)
    mse, mae, rmse, r2 = performance_metrics_regression(y_test, y_pred)
    print(f"mse: {mse} \nmae: {mae} \nrmse: {rmse} \nr2: {r2}")

    model.save_model("runs/MLP_Regression_best_model.pkl")


def pima_indians_diabetes():
    data_path = "../../data/processed/diabetes/diabetes_processed.csv"
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(-1, 1)

    # print(X.shape, y.shape)
    # print(X[0], y[0])

    val_split = 0.2
    
    X, X_val = np.split(X, [int((1-val_split)*X.shape[0])])
    y, y_val = np.split(y, [int((1-val_split)*y.shape[0])])

    l_r = 0.001
    activation_function = 'sigmoid'
    optimizer = 'mini_batch_gradient_descent'
    hidden_layers = 2
    neurons_per_layer = [16, 8]
    batch_size = 32
    no_of_epochs = 400
    input_layer_size = X.shape[1]
    output_layer_size = 1

    #regression with last layer activation function as sigmoid
    mlp = MLP(l_r=l_r,
                activation_function=activation_function,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                neurons_per_layer=neurons_per_layer,
                batch_size=batch_size,
                no_of_epochs=no_of_epochs,
                input_layer_size=input_layer_size,
                output_layer_size=output_layer_size)

    regression_losses = mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
    regression_y_pred = mlp.predict(X_val)
    
    #classification with last layer activation function as sigmoid
    from models.MLP.mlp_classification import MLP as MLP_classification
    mlp = MLP_classification(l_r=l_r,
                activation_function=activation_function,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                neurons_per_layer=neurons_per_layer,
                batch_size=batch_size,
                no_of_epochs=no_of_epochs,
                input_layer_size=input_layer_size,
                output_layer_size=2)
    
    classification_losses = mlp.fit(X, y, val=True, X_val=X_val, y_val=y_val, early_stopping=True, patience=5, wandb_log=False)
    classification_y_pred = mlp.predict(X_val)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(regression_losses, label="MSE")
    ax[0].set_title("MSE Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss(val)")
    ax[0].legend()

    ax[1].plot(classification_losses, label="BCE")
    ax[1].set_title("BCE Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss(val)")
    ax[1].legend()

    plt.savefig("figures/pima_indians_diabetes_loss_2.png")


def analysis_of_regression():
    with open("runs/MLP_Regression_best_hyperparams.json", "r") as file:
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
        output_layer_size=1
    )

    model.fit(X, y, early_stopping=True, patience=5, wandb_log=False)
    y_pred = model.predict(X_test)
    mse, mae, rmse, r2 = performance_metrics_regression(y_test, y_pred)
    print(f"mse: {mse} \nmae: {mae} \nrmse: {rmse} \nr2: {r2}")


    #plot the mse for each point in the test set
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(y_test, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.savefig("figures/Regression_analysis_actual_vs_predicted.png")    
    

    mse_per_point = (y_test - y_pred) ** 2
    median_mse = np.median(mse_per_point)

    low_mse_indices = np.where(mse_per_point <= median_mse)[0]
    high_mse_indices = np.where(mse_per_point > median_mse)[0]

    avg_low_mse_features = np.mean(X_test[low_mse_indices], axis=0)
    avg_high_mse_features = np.mean(X_test[high_mse_indices], axis=0)
    
    X_low_mse = X_test[low_mse_indices]
    X_high_mse = X_test[high_mse_indices]

    low_mse_df = pd.DataFrame(X_low_mse, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
    high_mse_df = pd.DataFrame(X_high_mse, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])

    low_mse_df["MSE_Group"] = "Low MSE"
    high_mse_df["MSE_Group"] = "High MSE"

    combined_df = pd.concat([low_mse_df, high_mse_df])
    
    fig, axes = plt.subplots(nrows=1, ncols=X_test.shape[1], figsize=(20, 5), sharey=True)
    for i, ax in enumerate(axes):
        sns.violinplot(x="MSE_Group", y=f"Feature_{i}", data=combined_df, ax=ax)
        ax.set_title(f"Feature {i}")

    plt.tight_layout()
    plt.savefig("figures/Regression_analysis_Violin_Low_vs_High_MSE.png")


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(avg_low_mse_features, label="Low MSE Avg Features", marker='o')
    ax.plot(avg_high_mse_features, label="High MSE Avg Features", marker='x')
    ax.set_title("Average Feature Values for Low and High MSE Points")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Average Feature Value")
    ax.legend()
    plt.savefig("figures/Regression_analysis_Low_vs_High_MSE_avg.png")




testing_mlp_regression()
# hyperparameter_tuning()
# generate_table()
# get_best_model()
# get_performance_of_best_model()
# pima_indians_diabetes()
# analysis_of_regression()


