import pandas as pd
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt

path_to_data = "../external/HousingData.csv"

df = pd.read_csv(path_to_data)

print(df.head())

print(df.isnull().sum())

# replace NA values with mean
for column in df.columns:
    df[column].fillna(df[column].mean(), inplace=True)

stats = []
columns = df.columns
for i in range(len(df.columns)):
    average = np.mean(df.iloc[:, i]).round(3)
    median = np.median(df.iloc[:, i]).round(3)
    std = np.std(df.iloc[:, i]).round(3)
    min = np.min(df.iloc[:, i]).round(3)
    max = np.max(df.iloc[:, i]).round(3)
    stats.append([columns[i], average, median, std, min, max])

stats_df = pd.DataFrame(stats, columns=["Feature", "Average", "Median", "Standard Deviation", "Min", "Max"])

x = pt.PrettyTable()
x.field_names = stats_df.columns

for i in range(len(stats_df)):
    x.add_row(stats_df.iloc[i])

x.set_style(pt.MARKDOWN)
print(x)

fig, ax = plt.subplots(7, 2, figsize=(6, 12))
fig.tight_layout(pad=3.0)
ax = ax.ravel()

for i in range(len(df.columns)):
    ax[i].violinplot(df.iloc[:, i], vert=False)
    ax[i].set_title(columns[i])

plt.savefig("../../assignments/3/figures/housing_distribution.png")


for column in df.columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df.to_csv("../../data/processed/HousingData/HousingData_normalised.csv", index=False)

