from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import re
from nltk.corpus import stopwords



stop = stopwords.words('english')
def tokenizer(text):
    """ Clean unprocessed text data from the move_data.csv 
    and separate it into word tokens while removing stop words
    
    :param text: string
        single review record 
    :return: string
        processed review record
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    """ Read one review and its sentiment at a time

    :param path: string
        path to the csv file
    :return: string and int
        string is a review itself and int is a label(0 => negative or 1 => positive)
    """
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    """ Get a specific number of records from the stream

    :param doc_stream: string (generator)
        record stream from the stream_docs function
    :param size: int
        number of records specified
    :return: array of strings, array of ints
        array of strings contains nubmer of records
        array of ints contains correspoding number of labels
    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

vect = HashingVectorizer(decode_error='ignore',
                         tokenizer=tokenizer,  # Callable tokenizer function
                         n_features=2**21)  # Reduce hash collisions
clf = SGDClassifier(loss='log', random_state=1, max_iter=1, tol=1e-3)
doc_stream = stream_docs(path='movie_data.csv')

# OUT-OF-CORE LEARNING
classes = np.array([0, 1])
# Allocate 45000 records for the training
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)

# Allocate 5000 records for the testing
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))