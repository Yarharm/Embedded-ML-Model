import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from vectorizer import vect

def update_model(db_path, model, batch_size=10000):
    """ Update movie classification model using new reviews enterred by the users

    :param db_path: string
        Path to the SQLite database
    :param model: model
        SGDClassifier
    :param batch_size: int
        Update model each 10000 reviews in the database
    :return: newly trained model
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model

cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
                  'pkl_objects',
                  'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

clf = update_model(db_path=db, model=clf, batch_size=10000)

# Serialize back newly trained model
pickle.dump(clf, open(os.path.join(cur_dir,
            'pkl_objects', 'classifier.pkl'), 'wb'), protocol=4)
