import numpy as np
import pandas as pd

data_file = "../external/word_embeddings.feather"
df = pd.read_feather(data_file)

# print(df['words'].unique().shape)

csv_file_path = "../processed/word_embeddings/word_embeddings.csv"

no_of_rows = df.shape[0]

with open(csv_file_path, "w") as f:
    f.write("word," + ",".join([str(x) for x in range(512)]) + "\n")

for i in range(no_of_rows):
    word = df['words'][i]
    vit = df['vit'][i]
    # print(vit.shape)
    vit_str = ",".join([str(x) for x in vit])

    with open(csv_file_path, "a") as f:
        f.write(word + "," + vit_str + "\n")
