"""
This script does basic data visualization for the regularisation.csv.
"""

train_path = '../../../data/processed/regularisation/train.csv'

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv(train_path)

x_train = train['x']
y_train = train['y']


plt.scatter(x_train, y_train, color='blue', label='train', s=10)
plt.legend()

plt.xlabel('x')
plt.ylabel('y')

plt.title("Distribution of Train set for Regularisation")

plt.savefig('../figures/lin_reg/regularisation.png')



