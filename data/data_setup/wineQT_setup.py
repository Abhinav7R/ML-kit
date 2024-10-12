import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import prettytable as pt


path_to_data = "../external/wineQT.csv"
df = pd.read_csv(path_to_data)

# print(df.head())
# print(df.describe())

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

# violin plot for each feature

fig, ax = plt.subplots(4, 3, figsize=(12, 9))
fig.tight_layout(pad=3.0)
ax = ax.ravel()

for i in range(len(df.columns)-1):
    ax[i].violinplot(df.iloc[:, i], vert=False)
    ax[i].set_title(columns[i])

plt.savefig("../../assignments/3/figures/wine_distribution.png")

print(df.isnull().sum())

#min max normalisation

for column in df.columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df.drop(columns=["Id"], inplace=True)

csv_path = "../processed/wineQT/wine_normalised.csv"
df.to_csv(csv_path, index=False)
