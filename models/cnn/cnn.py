"""
CNN class - inherited from torch.nn.Module
"""


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN class - inherited from torch.nn.Module
    Hnadles regression and classification tasks.
    Contains convolutional layers and fully connected layers.
    Also contains forward function to define the forward pass.
    """
    def __init__(self, task='classification', dropout_prob=0.5):
        super(CNN, self).__init__()

        if task != 'classification' and task != 'regression':
            raise ValueError("Task must be either 'classification' or 'regression'")

        self.task = task
        self.dropout_prob = dropout_prob

        # architecture 1
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 64x64 -> 32x32 after pooling
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 32x32 -> 16x16 after pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 16x16 -> 8x8 after pooling

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc1_drop = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(in_features=128, out_features=1 if task == 'regression' else 4)
        # 1 output for regression, 4 for classification (0, 1, 2, 3)

        #----------------------------------------------#

        # # architecture 2
        # # Convolutional layers
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 64x64 -> 32x32 after pooling
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 32x32 -> 16x16 after pooling
        # # Fully connected layer
        # self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=64)
        # self.fc1_drop = nn.Dropout(p=dropout_prob)
        # self.fc2 = nn.Linear(in_features=64, out_features=1 if task == 'regression' else 4)

        #----------------------------------------------#


    def forward(self, x):
        # conv1 -> relu -> maxpool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # conv2 -> relu -> maxpool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # comment out the following lines for architecture 2
        # conv3 -> relu -> maxpool
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # fc1 -> relu -> fc2
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # NOTE: no softmax for classification, as we are using CrossEntropyLoss
        # softmax is included in the loss function

        return x

        