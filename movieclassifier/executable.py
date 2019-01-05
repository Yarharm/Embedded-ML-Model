import pickle
import os
import numpy as np
from movieclassifier.vectorizer import vect

clf = pickle.load(open(
        os.path.join('pkl_objects',
                      'classifier.pkl'), 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['Great movie']
example_vectorized = vect.transform(example)

print('Prediction: %s\nProbability: %.2f%%' %\
      (label[clf.predict(example_vectorized)[0]],
       np.max(clf.predict_proba(example_vectorized)) * 100))