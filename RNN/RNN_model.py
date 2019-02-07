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
padded_sequence_len = 200
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

# RNN model
class RNN(object):
    def __init__(self, n_words, seq_len=200, lstm_size=256, num_layers=1,
                 batch_size=64, learning_rate=0.0001, embed_size=200):
        ## Sequence length and Embedding size are identical in this implementation
        self.n_words = n_words  # numb of unique words
        self.seq_len = seq_len  # length of the transformed review(Left-padded)
        self.lstm_size = lstm_size  # numb of units in a hidden layer
        self.num_layers = num_layers  # numb of RNN layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size  # embedding layers (number of features)

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(1)
            self.build()
            self.saver = tf.train.Saver()  # Saving is optional
            self.init_op = tf.global_variables_initializer()

    def build(self):
        """
        Declares three placeholdres: input data, labels and probability of drop out layer
        Create embedding layer and build RNN using embedded feature representation
        :return:
        """
        # Placeholders
        tf_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_len),
                              name='tf_x')
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name='tf_y')
        tf_keep_proba = tf.placeholder(tf.float32, name='tf_keep_proba')

        # Embedding layer (Integer encoded review => embedding feature vector)
        embed_lay = tf.Variable(tf.random_uniform((self.n_words, self.embed_size),  # Real numbers [-1,1]
                                                  minval=-1, maxval=1), name='embed_lay')
        embed_x = tf.nn.embedding_lookup(embed_lay, tf_x, name='embed_x')  #look up for embedding matrix

        # Complex cells with Dropout on top of LSTM (Numb of cells is self.num_layers)
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.lstm_size),
                                                                           output_keep_prob='tf_keep_proba')
                                             for i in range(self.num_layers)])

        # States of the cell
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)

        ## Create RNN defined by the cells specifications
        # Perform fully dynamic unrolling of inputs
        # self.final_state shape [batch_size, self.num_layers]
        # lstm_outputs shape [batch_size, max_time, self.num_layers]
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, embed_x, initial_state=self.initial_state)

        # Unprocessed final fully connected layer
        logits = tf.layers.dense(inputs=lstm_outputs[:, -1],
                                 units=1, activation=None,
                                 name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')

        # Apply activ func on final layer to get probas and labels
        y_proba = tf.nn.sigmoid(logits, name='probabilities')
        labels = tf.cast(tf.round(y_proba), tf.int32, name='labels')
        predictions = {'probabilities': y_proba,
                       'labels': labels}

        # Cost and Optimizer
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labrls=tf_y, logits=logits),
                              name='cost')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

    def train(self, X_train, y_train, num_epochs):
        """
        At the start of each epoch, state of the cell is zero
        During each epoch, data is pushed along side with the current state
        At the end of the epoch, state is updated
        :param X_train: Input data
        :param y_train: Target data
        :param num_epochs: Number of epochs
        :return: None
        """
        with tf.Session(graph=self.g) as session:
            session.run(self.init_op)  # Init variables
            iteration = 1
            for epoch in range(num_epochs):
                state = session.run(self.initial_state)
                for batch_x, batch_y in batch_generator(X_train, y_train, self.batch_size):
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keep_proba:0': 0.5,
                            self.initial_state: state}
                    # Uodate state by the end of the epoch
                    loss, _, state = session.run(['cost:0', 'train_op', self.final_state],
                                                 feed_dict=feed)

#    def predict(self, X_test):
#        with tf.Session(graph=g) as session:
