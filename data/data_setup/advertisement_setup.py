import pandas as pd
import numpy as np

path_to_data = "../external/advertisement.csv"

df = pd.read_csv(path_to_data)

print(df.head())

#drop the city column
df = df.drop('city', axis=1)

# find unique values in columns
# df['occupation'].unique()
# df['education'].unique()
# df['most bought item'].unique()

#label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['occupation'] = le.fit_transform(df['occupation'])
df['gender'] = le.fit_transform(df['gender'])
df['married'] = le.fit_transform(df['married'])
df['education'] = le.fit_transform(df['education'])
df['most bought item'] = le.fit_transform(df['most bought item'])

print(df.head())

labels = df['labels']
# multiple labels exist in the labels column
# find unique values in labels
label_list = []
for label in labels:
    label_list.extend(label.split(' '))

label_list = list(set(label_list))
print(label_list)

#noramlaise the labels
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['age'] = scaler.fit_transform(df[['age']])
df['income'] = scaler.fit_transform(df[['income']])
df['education'] = scaler.fit_transform(df[['education']])
df['occupation'] = scaler.fit_transform(df[['occupation']])
df['most bought item'] = scaler.fit_transform(df[['most bought item']])
df['gender'] = scaler.fit_transform(df[['gender']])
df['married'] = scaler.fit_transform(df[['married']])
df['purchase_amount'] = scaler.fit_transform(df[['purchase_amount']])

save_advertisement_path = "../processed/advertisement/advertisement.csv"
df.to_csv(save_advertisement_path, index=False)
