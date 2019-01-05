# Embedded-ML-model-in-Web-App

**Sentiment analysis** with embedding movie classifier model into a web application.

Large Movie Review [Dataset](https://nlp.stanford.edu/~amaas/data/sentiment/).
- 25,000 highly polar movie reviews for training.
- 25,000 reviews for testing.

Code for conversion between aclImdb(txt files) to a single csv file is in _txt_to_csv_script_.

[Dataset copyright details](https://nlp.stanford.edu/~amaas/papers/wvSent_acl2011.bib).


**out-of-core learning** with SGDClassifier and train a logistic regression model using mini-batches of documents.

Web app is build upon Flask framework, processed movie reviews are stored in SQLite database and used for the future model training.
