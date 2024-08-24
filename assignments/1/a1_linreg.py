"""
This script is the main script that uses the Linear Regression model.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..','..')))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from models.linear_regression.linear_regression import Linear_Regression
from performance_measures.performance_measures import Performance_Measures

path_to_data = "../../data/processed/lin_reg"
train = pd.read_csv(f'{path_to_data}/train.csv')
test = pd.read_csv(f'{path_to_data}/test.csv')
val = pd.read_csv(f'{path_to_data}/val.csv')

X_train = train.drop(columns=['y']).values
y_train = train['y'].values

X_val = val.drop(columns=['y']).values
y_val = val['y'].values

X_test = test.drop(columns=['y']).values
y_test = test['y'].values

def simple_LR_on_train_test_val():
    model = Linear_Regression(learning_rate=0.01, epochs=10000, lambda_=0, closed_form=False, regularization=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    perf_measures = Performance_Measures()

    mse = perf_measures.mean_square_error(y_val, y_pred)
    std = perf_measures.standard_deviation(y_val, y_pred)
    var = perf_measures.variance(y_val, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Standard Deviation: {std}")
    print(f"Variance: {var}")

    with open("results/lin_reg_results.txt", "a") as f:
        f.write("Linear Regression results on val set\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Standard Deviation: {std}\n")
        f.write(f"Variance: {var}\n")
        f.write("\n")

def compare_with_sklearn():

    from sklearn.linear_model import Ridge, Lasso, LinearRegression as Li
    from sklearn.metrics import mean_squared_error

    model = Li()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    perf_measures = Performance_Measures()
    mse = mean_squared_error(y_val, y_pred)
    var = perf_measures.variance(y_val, y_pred)
    std = perf_measures.standard_deviation(y_val, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Standard Deviation: {std}")
    print(f"Variance: {var}")

def plot_line_of_fit_and_train_points():

    model = Linear_Regression(learning_rate=0.01, epochs=10000, lambda_=0, closed_form=False, regularization=None)
    model.fit(X_train, y_train)

    array_of_x_values = pd.DataFrame(np.linspace(-1.2, 1.2, 100)).values

    y_pred = model.predict(array_of_x_values)

    plt.scatter(X_train, y_train, color='blue', label='train', s=10)
    plt.plot(array_of_x_values, y_pred, color='red', label='line of fit')
    plt.legend()
    plt.title("Line of fit and train points")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('figures/lin_reg/line_of_fit_and_train.png')

def play_with_learning_rates():
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    mse = []
    for lr in learning_rates:
        model = Linear_Regression(learning_rate=lr, epochs=10000, lambda_=0, closed_form=False, regularization=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse.append(Performance_Measures().mean_square_error(y_val, y_pred))

    plt.plot(learning_rates, mse, color='blue', marker='o')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Mean Squared Error')
    plt.xscale('log')
    plt.title('Mean Squared Error vs Learning Rate (gradient descent)')
    plt.savefig('figures/lin_reg/mse_vs_lr_gradient.png')

    print(min(mse))
    print(learning_rates[mse.index(min(mse))])

def degree_greater_than_1():
    with open("results/lin_reg_higher_degree.txt", "w") as f:
        f.write("Results for Linear Regression with higher degree polynomials\n")
    mse_test_list = []
    max_k = 30
    for k in range(1,max_k):
        model = Linear_Regression(learning_rate=0.1, epochs=10000, lambda_=0, closed_form=False, regularization=None, k=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse_test = Performance_Measures().mean_square_error(y_test, y_pred)
        var_test = Performance_Measures().variance(y_test, y_pred)
        std_test = Performance_Measures().standard_deviation(y_test, y_pred)
        mse_test_list.append(mse_test)

        y_pred = model.predict(X_train)
        mse_train = Performance_Measures().mean_square_error(y_train, y_pred)
        var_train = Performance_Measures().variance(y_train, y_pred)
        std_train = Performance_Measures().standard_deviation(y_train, y_pred)

        params = model.get_params()

        with open("results/lin_reg_higher_degree.txt", "a") as f:
            f.write(f"Polynomial Degree: {k}\n")
            f.write("Train \n\n")
            f.write(f"Mean Squared Error: {mse_train}\n")
            f.write(f"Standard Deviation: {std_train}\n")
            f.write(f"Variance: {var_train}\n")
            f.write("Test \n\n")
            f.write(f"Mean Squared Error: {mse_test}\n")
            f.write(f"Standard Deviation: {std_test}\n")
            f.write(f"Variance: {var_test}\n")
            f.write(f"Parameters: {params}\n")

    plt.plot([i for i in range(1,max_k)], mse_test_list, color='blue', marker='o', markersize=4)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error on Test Set')
    plt.title('Mean Squared Error (Test) vs Polynomial Degree (gradient descent)')
    plt.savefig('figures/lin_reg/mse_vs_degree_gradient_descent.png')

    print("Minimum MSE: ", min(mse_test_list))
    print("Polynomial Degree with min MSE on test set: ", mse_test_list.index(min(mse_test_list)) + 1)

def save_images_for_gif():
    model = Linear_Regression(learning_rate=0.01, epochs=250, lambda_=0, closed_form=False, regularization=None, k=15)
    model.fit(X_train, y_train)

def regularisation_analysis():

    train = pd.read_csv('../../data/processed/regularisation/train.csv')
    test = pd.read_csv('../../data/processed/regularisation/test.csv')

    X_train = train.drop(columns=['y']).values
    y_train = train['y'].values

    X_test = test.drop(columns=['y']).values
    y_test = test['y'].values

    fig, ax = plt.subplots(4, 5, figsize=(30, 25))
    ax = ax.ravel()

    max_k = 21
    mse_test_list = []
    for k in range(1,max_k):
        model = Linear_Regression(learning_rate=0.1, epochs=10000, lambda_=5, closed_form=False, regularization='l2', k=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse_test = Performance_Measures().mean_square_error(y_test, y_pred)
        var_test = Performance_Measures().variance(y_test, y_pred)
        std_test = Performance_Measures().standard_deviation(y_test, y_pred)

        X_test_sorted = np.sort(X_test, axis=0)
        y_pred_sorted = model.predict(X_test_sorted)

        results = f"MSE: {mse_test.round(2)}\nVariance: {var_test.round(2)}\nStandard Deviation: {std_test.round(2)}"
        
        ax[k-1].scatter(X_train, y_train, color='blue', label='train', s=10)
        ax[k-1].plot(X_test_sorted, y_pred_sorted, color='red', label='line of fit')
        ax[k-1].set_title(f"Degree: {k}")
        ax[k-1].text(0.2, 0.95, results, transform=ax[k-1].transAxes, fontsize=12, verticalalignment='top')

        mse_test_list.append([k, mse_test])

    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig('figures/lin_reg/l2_regularisation.png')
    plt.close()

    min_k = min(mse_test_list, key=lambda x: x[1])
    print("Minimum MSE: ", min_k[1])
    print("Polynomial Degree with min MSE on test set: ", min_k[0])

    #plot mse vs degree
    mse_test_list = np.array(mse_test_list)
    plt.plot(mse_test_list[:,0], mse_test_list[:,1], color='blue', marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error on Test Set')
    plt.title('MSE (Test) vs Degree L2 Regularisation')
    plt.savefig('figures/lin_reg/l2_reg_mse_vs_degree.png')
    
def regularisation_comparison():
    train = pd.read_csv('../../data/processed/regularisation/train.csv')
    test = pd.read_csv('../../data/processed/regularisation/test.csv')

    X_train = train.drop(columns=['y']).values
    y_train = train['y'].values

    X_test = test.drop(columns=['y']).values
    y_test = test['y'].values  

    mse_test_1 = []
    mse_test_2 = []
    mse_test_3 = []

    for k in range(1, 21):
        model1 = Linear_Regression(learning_rate=0.01, epochs=10000, lambda_=0, closed_form=False, regularization=None, k=k)
        model1.fit(X_train, y_train)
        y_pred = model1.predict(X_test)
        mse_test_1.append(Performance_Measures().mean_square_error(y_test, y_pred))

        model2 = Linear_Regression(learning_rate=0.01, epochs=10000, lambda_=1, closed_form=False, regularization='l1', k=k)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        mse_test_2.append(Performance_Measures().mean_square_error(y_test, y_pred))
                    
        model3 = Linear_Regression(learning_rate=0.01, epochs=10000, lambda_=1, closed_form=False, regularization='l2', k=k)
        model3.fit(X_train, y_train)
        y_pred = model3.predict(X_test)
        mse_test_3.append(Performance_Measures().mean_square_error(y_test, y_pred))

    plt.figure(figsize=(6, 6))
    plt.plot([i for i in range(1,21)], mse_test_1, color='blue', marker='o', label='No Regularisation', markersize=4)
    plt.plot([i for i in range(1,21)], mse_test_2, color='red', marker='o', label='L1 Regularisation', markersize=4)
    plt.plot([i for i in range(1,21)], mse_test_3, color='green', marker='o', label='L2 Regularisation', markersize=4)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MSE on Test Set')
    plt.title('MSE (Test) vs Degree Regularisation Comparison')
    plt.ylim(0, 0.3)
    plt.legend()    
    plt.savefig('figures/lin_reg/regularisation_comparison_mse.png')

# simple_LR_on_train_test_val()
# compare_with_sklearn()
# plot_line_of_fit_and_train_points()
# play_with_learning_rates()
# degree_greater_than_1()
# save_images_for_gif()
# regularisation_analysis()
# regularisation_comparison()
