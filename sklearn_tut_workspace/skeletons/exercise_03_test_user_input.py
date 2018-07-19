"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

sentence = [sys.argv[1]]

language_clf_pkl = open('language.pkl', 'rb')
language_clf = pickle.load(language_clf_pkl)

language_dataset_pkl = open('language_dataset.pkl', 'rb')
language_dataset = pickle.load(language_dataset_pkl)


sentiment_clf_pkl = open('sentiment.pkl', 'rb')
sentiment_clf = pickle.load(sentiment_clf_pkl)

sentiment_dataset_pkl = open('sentiment_dataset.pkl', 'rb')
sentiment_dataset = pickle.load(sentiment_dataset_pkl)

language = language_clf.predict(sentence)
language = language_dataset.target_names[language[0]]

if language == 'en':
    predicted = sentiment_clf.predict(sentence)
    confidence = sentiment_clf.predict_proba(sentence)

    sentiment, confidence = sentiment_dataset.target_names[predicted[0]], confidence[0][predicted]
else:
    sentiment, confidence = 'not available', float('nan')
print(u'The language of "%s" is %s, the sentiment is %s with probability %0.2f'
      % (sentence[0], language, sentiment, confidence))
