import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter
import tensorflow as tf

# Review data(Shape[50000x2] {review x sentiment})
df = pd.read_csv('movie_data.csv', low_memory=False)

# Get only unique words
counts = Counter()  # Dictionary {keys: words, values: their count}
for i, review in enumerate(df['review']):
    text = ''.join([char if char not in punctuation else ' '+char+' '
                   for char in review]).lower()  # create space between punctuations
    df.loc[i, 'review'] = text
    counts.update(text.split())

word_counts = sorted(counts, key=counts.get, reverse=True)  # unqiue words
#print(word_counts[:5])

# Map unique words to integers
word_to_int = {word: integ for integ, word in enumerate(word_counts, 1)}

mapped_reviews = []
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
#print(mapped_reviews[:5])


# Adjust review to have the same size
# Left-padding (Length = 200 => open to Optimization)
#print(np.array(mapped_reviews).shape)
padded_sequence_len = 256
sequences = np.zeros((df.shape[0], padded_sequence_len), dtype=int)  # zero matrix

for i, review in enumerate(mapped_reviews):
    review_arr = np.array(review)
    sequences[i, -len(review):] = review_arr[-padded_sequence_len:]  # Chop review if len > 200


# Get train/test data
X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].values

# Cross validation with mini-batches
np.random.seed(1)
def batch_generator(X, y=None, batch_size=64):
    """
    Get batch generator
    :param X: input data, shape {n_records x paddes_sequence_len}
    :param y: target data, shape {n_records}
    :param batch_size: size of the batch, int
    :return: batch size generated values
    """
    n_batches = len(X) // batch_size
    X = X[:n_batches * batch_size]  # Get even chunks
    if y is not None:
        y = y[:n_batches * batch_size]
    for ii in range(0, len(X), batch_size):
        if y is not None:
            yield X[ii:batch_size], y[ii:batch_size]
        else:
            yield X[ii:batch_size]
    return

## Word sequence must be converted to the input features
## One Hot Enconding (Bad solution: Matrix is too sparse)
## Embedding (Great solution)
