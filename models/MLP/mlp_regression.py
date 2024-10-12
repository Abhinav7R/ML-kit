"""
This is the MLP Regression model. It is implemented from scratch without using any libraries.
"""

import numpy as np
import wandb
import pickle

class MLP_Regression:
    def __init__(self,
                l_r=0.001,
                activation_function='linear',
                optimizer='sgd',
                hidden_layers=2,
                neurons_per_layer=[3, 4],
                batch_size=32,
                no_of_epochs=100,
                input_layer_size=None,
                output_layer_size=None
                ):

        self.X = None
        self.y = None

        self.l_r = l_r
        self.activation_functions = ['sigmoid', 'tanh', 'relu', 'linear']
        if activation_function not in self.activation_functions:
            raise ValueError("Activation function should be either 'sigmoid' or 'tanh' or 'relu' or 'linear'")
        self.activation_function = activation_function
        self.optimizers = ['sgd', 'batch_gradient_descent', 'mini_batch_gradient_descent']
        if optimizer not in self.optimizers:
            raise ValueError("Optimizer should be either 'sgd' or 'batch_gradient_descent' or 'mini_batch_gradient_descent'")
        self.optimizer = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

        if len(neurons_per_layer) != hidden_layers:
            raise ValueError("Length of neurons per layer must match the number of hidden layers")
        
        self.batch_size = batch_size

        self.no_of_epochs = no_of_epochs

        if input_layer_size is None or output_layer_size is None:
            raise ValueError("Input layer size and output layer size must be provided")

        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size

        self.weights = []
        self.biases = []

        layers = [self.input_layer_size] + self.neurons_per_layer + [self.output_layer_size]

        for i in range(len(layers)-1):
            weight_matrix = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            self.weights.append(weight_matrix)

            bias_vector = np.zeros((1, layers[i+1]))
            self.biases.append(bias_vector)

        self.activations = []
        self.z_values = []

        for i in range(len(layers)):
            self.activations.append(np.zeros((layers[i], 1)))
            self.z_values.append(np.zeros((layers[i], 1)))

        self.weight_gradients = [np.zeros_like(w) for w in self.weights]
        self.bias_gradients = [np.zeros_like(b) for b in self.biases]

        self.weight_gradients_numerical = [np.zeros_like(w) for w in self.weights]
        self.bias_gradients_numerical = [np.zeros_like(b) for b in self.biases]


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def linear(self, x):
        return x
    
    def linear_derivative(self, x):
        return np.ones_like(x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _apply_activation(self, x, function='sigmoid'):
        if function == 'sigmoid':
            return self.sigmoid(x)
        elif function == 'tanh':
            return self.tanh(x)
        elif function == 'relu':
            return self.relu(x)
        elif function == 'linear':
            return self.linear(x)
        
    def _apply_activation_derivative(self, x, function='sigmoid'):
        if function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif function == 'tanh':
            return self.tanh_derivative(x)
        elif function == 'relu':
            return self.relu_derivative(x)
        elif function == 'linear':
            return self.linear_derivative(x)
    
    def forward_pass(self, X):
        self.activations[0] = X
        self.z_values[0] = X
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values[i+1] = z
            self.activations[i+1] = self._apply_activation(z, self.activation_function)
        
        self.activations[-1] = self.linear(self.z_values[-1])
        # for bce loss comparison
        # self.activations[-1] = self.sigmoid(self.z_values[-1])
        return self.activations, self.z_values

    def back_propagation(self, X, y):
        y_pred = self.activations[-1]
        delta = y_pred - y
        # for bce loss comparison
        # delta = delta * self.sigmoid_derivative(self.z_values[-1])

        self.weight_gradients[-1] = np.dot(self.activations[-2].T, delta) / X.shape[0]
        self.bias_gradients[-1] = np.mean(delta, axis=0, keepdims=True)

        for i in range(len(self.weights)-1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self._apply_activation_derivative(self.z_values[i], self.activation_function)
            self.weight_gradients[i-1] = np.dot(self.activations[i-1].T, delta) / X.shape[0]
            self.bias_gradients[i-1] = np.array([np.mean(delta, axis=0)])

    def weight_update(self):

        for i in range(len(self.weights)):
            self.weight_gradients[i] = np.clip(self.weight_gradients[i], -1, 1)
            self.bias_gradients[i] = np.clip(self.bias_gradients[i], -1, 1)
            self.weights[i] -= self.l_r * self.weight_gradients[i]
            self.biases[i] -= self.l_r * self.bias_gradients[i]

    def fit(self, X, y, val=False, X_val=None, y_val=None, early_stopping=False, patience=5, wandb_log=False):
        self.X = X
        self.y = y

        if val:
            losses = []
            if X_val is None or y_val is None:
                raise ValueError("Validation data must be provided")
            if early_stopping:
                patience_counter = 0

        for epoch in range(self.no_of_epochs):
            if self.optimizer == 'batch_gradient_descent':
                self.forward_pass(X)
                self.back_propagation(X, y)
                # self.check_gradients(X, y)
                self.weight_update()
            elif self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    self.forward_pass(X[i].reshape(1, -1))
                    self.back_propagation(X[i].reshape(1, -1), y[i].reshape(1, -1))
                    # self.check_gradients(X[i].reshape(1, -1), y[i].reshape(1, -1))
                    self.weight_update()
            elif self.optimizer == 'mini_batch_gradient_descent':
                for i in range(0, X.shape[0], self.batch_size):
                    self.forward_pass(X[i:i+self.batch_size])
                    self.back_propagation(X[i:i+self.batch_size], y[i:i+self.batch_size])
                    # self.check_gradients(X[i:i+self.batch_size], y[i:i+self.batch_size])
                    self.weight_update()
            error = self.cost_function(X, y)
            # print(f"Epoch: {epoch+1}/{self.no_of_epochs}, Error: {error}")

            if val:
                
                error_val = self.cost_function(X_val, y_val)
                # print(f"Validation Error: {error_val}")
                losses.append(error_val)
                if early_stopping:
                    if epoch == 0:
                        best_error = error_val
                        best_weights = self.weights
                        best_biases = self.biases
                    else:
                        if error_val < best_error:
                            best_error = error_val
                            best_weights = self.weights
                            best_biases = self.biases
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter == patience:
                                # print(f"Early stopping at epoch: {epoch+1}")
                                self.weights = best_weights
                                self.biases = best_biases
                                break
                if wandb_log:
                    y_pred = self.predict(X_val)
                    val_mse = self.mse(y_val, y_pred)
                    val_rmse = self.rmse(y_val, y_pred)
                    val_r_squared = self.r_squared(y_val, y_pred)
                    y_pred_train = self.predict(X)
                    train_mse = self.mse(y, y_pred_train)
                    train_rmse = self.rmse(y, y_pred_train)
                    train_r_squared = self.r_squared(y, y_pred_train)
                    wandb.log({
                    'epoch': epoch,
                    'train_loss': error,
                    'val_loss': error_val,
                    'val_mse': val_mse,
                    'val_rmse': val_rmse,
                    'val_r_squared': val_r_squared,
                    'train_mse': train_mse,
                    'train_rmse': train_rmse,
                    'train_r_squared': train_r_squared
                    })
        if val:
            return losses
        
            
    def predict(self, X):
        self.forward_pass(X)
        return self.activations[-1]

    def cost_function(self, X, y_true):
        activations, _ = self.forward_pass(X)
        y_pred = activations[-1]
        # MSE
        return np.mean((y_true - y_pred)**2)
    
    def numerical_gradients(self, X, y):
        epsilon = 1e-5
        
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += epsilon
                    error_plus = self.cost_function(X, y)
                    self.weights[i][j, k] -= 2*epsilon
                    error_minus = self.cost_function(X, y)
                    self.weights[i][j, k] += epsilon
                    self.weight_gradients_numerical[i][j, k] = (error_plus - error_minus) / (2*epsilon)
        
        for i in range(len(self.biases)):
            for j in range(self.biases[i].shape[1]):
                self.biases[i][0, j] += epsilon
                error_plus = self.cost_function(X, y)
                self.biases[i][0, j] -= 2*epsilon
                error_minus = self.cost_function(X, y)
                self.biases[i][0, j] += epsilon
                self.bias_gradients_numerical[i][0, j] = (error_plus - error_minus) / (2*epsilon)
        
        return self.weight_gradients_numerical, self.bias_gradients_numerical
           
    def compare_gradients(self):
        # value = norm(grad - grad_num) / (norm(grad +grad_num)

        for i in range(len(self.weight_gradients)):
            grad = self.weight_gradients[i]
            grad_num = self.weight_gradients_numerical[i]
            diff = np.linalg.norm(grad - grad_num) / (np.linalg.norm(grad + grad_num))
            print(f"Weight Gradient Difference: {diff}")
            # print(grad)
            # print(grad_num)
              
        for i in range(len(self.bias_gradients)):
            grad = self.bias_gradients[i]
            grad_num = self.bias_gradients_numerical[i]
            diff = np.linalg.norm(grad - grad_num) / (np.linalg.norm(grad + grad_num))
            print(f"Bias Gradient Difference: {diff}")
            # print(grad)
            # print(grad_num)
            
    def check_gradients(self, X, y):
        self.numerical_gradients(X, y)
        self.compare_gradients()
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def r_squared(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
        # from sklearn.metrics import r2_score
        # return r2_score(y_true, y_pred)

    def get_params(self):
        params = {
            "weights": self.weights,
            "biases": self.biases,
        }
        return params
    
    def save_model(self, path):
        params = self.get_params()
        with open(path, 'wb') as f:
            pickle.dump(params, f)