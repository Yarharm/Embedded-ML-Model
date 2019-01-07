from sklearn.feature_extraction.text import HashingVectorizer
import re
import pickle
import os

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
    os.path.join(cur_dir,
                 'pkl_objects',
                 'stopwords.pkl'), 'rb'))

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


vect = HashingVectorizer(decode_error='ignore',
                         tokenizer=tokenizer,
                         n_features=2**21)