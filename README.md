# Embedded-ML-model-in-Web-App

**Sentiment analysis** with embedding movie classifier model into a web application.

Large Movie Review [Dataset](https://nlp.stanford.edu/~amaas/data/sentiment/).
- 25,000 highly polar movie reviews for training.
- 25,000 reviews for testing.

Code for conversion between aclImdb(txt files) to a single csv file is in _txt_to_csv_script_.

[Dataset copyright details](https://nlp.stanford.edu/~amaas/papers/wvSent_acl2011.bib).


**out-of-core learning** using SGDClassifier with logistic regression model using mini-batches of documents (stochastic gradient descent).

Web application allows users to insert a minimum 15 characters review about any movie and then identifies whether the review is positive or negative. 

Application is build upon Flask framework, processed movie reviews are stored in SQLite database and used for the future model training.

__Styling in proccess__

__Deployment in proccess__ (Possibly with [PythonAnywhere](https://www.pythonanywhere.com/))