"""
This is the Auto Encoder model. It is implemented from scratch without using any libraries.
"""

import numpy as np

from models.MLP.mlp import MLP

class AutoEncoder:
    def __init__(self,
                l_r = 0.001,
                activation_function='sigmoid',
                optimizer='mini_batch_gradient_descent',
                hidden_layers=1,
                neurons_per_layer=[8],
                reduced_dimension=4,
                batch_size=32,
                no_of_epochs=100,
                input_layer_size=None):
        
        self.l_r = l_r
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.reduced_dimension = reduced_dimension
        self.batch_size = batch_size
        self.no_of_epochs = no_of_epochs
        self.input_layer_size = input_layer_size

        # hidden layers are layers without input and output layers
        self.hidden_layers = neurons_per_layer + [reduced_dimension] + neurons_per_layer[::-1]

        self.regressor = MLP(l_r=l_r,
                            activation_function=activation_function,
                            optimizer=optimizer,
                            hidden_layers=len(self.hidden_layers),
                            neurons_per_layer=self.hidden_layers,
                            batch_size=batch_size,
                            no_of_epochs=no_of_epochs,
                            input_layer_size=input_layer_size,
                            output_layer_size=input_layer_size,
                            task='regression')

    def fit(self, X, val=False, X_val=None, early_stopping=False, patience=5, wandb_log=False):
        if val and X_val is None:
            raise ValueError("Validation data is not provided")
        if val:            
            self.regressor.fit(X, X, val=val, X_val=X_val, y_val=X_val, early_stopping=early_stopping, patience=patience, wandb_log=wandb_log)
        else:
            losses = self.regressor.fit(X, X, val=val, X_val=X_val, y_val=X_val, early_stopping=early_stopping, patience=patience, wandb_log=wandb_log)
            return losses

    def get_latent(self, X):
        activations, _ = self.regressor.forward_pass(X)
        total_layers = len(self.hidden_layers) + 2 # input and output layers
        return activations[total_layers // 2]
    
    def reconstruction_loss(self, X):
        y_pred = self.regressor.predict(X)
        return np.mean((X - y_pred) ** 2)
