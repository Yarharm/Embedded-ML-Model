from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# Import HashingVectorizer
from vectorizer import vect
app = Flask(__name__)

# PREPARE CLASSIFIER
cur_dir = os.path.dirname(__file__)

# Deserialize Classifier
clf = pickle.load(open(
        os.path.join(cur_dir,
                     'pkl_objects',
                     'classifier.pkl'), 'rb'))

# Path to database


db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    """ Classify movie review to either positive(1) or negative(0)

    :param document: string
        Movie review
    :return: string, float
        string => classification label
        float => probability of the prediction
    """
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    """ Update classifier

    :param document: string
        Movie review
    :param y: int (0 or 1)
        Correct classification label
    :return: None
    """
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    """ Store submitted review, classification label and the timestamp in the database

    :param path: String
        Path to the database
    :param document: String
        Movie review
    :param y: int (0 or 1)
        Correct classification label
    :return: None
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, data)"\
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

# PREPATE FLASK
class ReviewForm(Form):
    """ Instantiate a TextAreaField, which is rendered in reviewform.html template
    """
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                 validators.length(min=15)])  # Review contains at least 15 char
@app.route('/')
def index():
    """
    Render reviewform.html (landing page of the app)
    """
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def result():
    """ Fetch the contents of the submitted form
    and pass it to the classifier

    :return: form
        Displa result of the prediction in the render results.html
    """
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba * 100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    """ Fetch predicted result from the results.html
    and add an entry to the database

    :return: form
        Render thanks.html to thank the user for the feedback
    """
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not (y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run()

