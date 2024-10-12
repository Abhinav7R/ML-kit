import pandas as pd
import numpy as np

path_to_data = "../external/diabetes.csv"

df = pd.read_csv(path_to_data)

print(df.isna().sum())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Pregnancies'] = scaler.fit_transform(df[['Pregnancies']])
df['Glucose'] = scaler.fit_transform(df[['Glucose']])
df['BloodPressure'] = scaler.fit_transform(df[['BloodPressure']])
df['SkinThickness'] = scaler.fit_transform(df[['SkinThickness']])
df['Insulin'] = scaler.fit_transform(df[['Insulin']])
df['BMI'] = scaler.fit_transform(df[['BMI']])
df['DiabetesPedigreeFunction'] = scaler.fit_transform(df[['DiabetesPedigreeFunction']])
df['Age'] = scaler.fit_transform(df[['Age']])
# df['Outcome'] = scaler.fit_transform(df[['Outcome']])

save_diabetes_path = "../processed/diabetes/diabetes_processed.csv"

df.to_csv(save_diabetes_path, index=False)

