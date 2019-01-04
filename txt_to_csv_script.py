import pandas as pd
import numpy as np
import os

# Path to files
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}

df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)

df.columns = ['review', 'sentiment']

# Shuffle and convert to csv
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# Check
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))