"""
This script does basic data visualization for the linear regression model dataset.
"""

train_path = '../../data/processed/lin_reg/train.csv'
test_path = '../../data/processed/lin_reg/test.csv'
val_path = '../../data/processed/lin_reg/val.csv'

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
val = pd.read_csv(val_path)

x_train = train['x']
y_train = train['y']

x_test = test['x']
y_test = test['y']

x_val = val['x']
y_val = val['y']

plt.scatter(x_train, y_train, color='blue', label='train', s=10)
plt.scatter(x_test, y_test, color='red', label='test', s=10)
plt.scatter(x_val, y_val, color='green', label='val', s=10)
plt.legend()

plt.xlabel('x')
plt.ylabel('y')

plt.title("Distribution of Train, Test and Val sets for Linear Regression")

plt.savefig('../figures/lin_reg/train_test_val.png')



