import pandas as pd
from string import punctuation
from collections import Counter


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

### TO DO:
# Review sequences are converted to the integers,
# but they have a different length (RNN architecture requirement).
# POSSIBLE SOLUTION: Left Padding