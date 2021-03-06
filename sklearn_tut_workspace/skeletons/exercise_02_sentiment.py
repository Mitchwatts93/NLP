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



if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([('vect', TfidfVectorizer(max_df= 0.95, min_df = 3, ngram_range=(1,1))),
                             ('clf', SVC(kernel = 'linear',C=1000, probability=True))])

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    #paramters = { 'vect__ngram_range': [(1,3),(1,4)]}

    #gs_sclf = GridSearchCV(pipeline, paramters, n_jobs = -1)
    gs_sclf = pipeline.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    #n_candidates = len(gs_sclf.cv_results_['params'])
    #print('n_candidates', n_candidates)
    #for i in range(n_candidates):
    #    print(i, 'params - %s; mean - %0.2f; std - %0.2f'
    #             % (gs_sclf.cv_results_['params'][i],
    #                gs_sclf.cv_results_['mean_test_score'][i],
    #                gs_sclf.cv_results_['std_test_score'][i]))


    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = gs_sclf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    sentiment_pkl = open('sentiment.pkl', 'wb')
    pickle.dump(gs_sclf, sentiment_pkl)
    sentiment_pkl.close()

    sentiment_dataset_pkl = open('sentiment_dataset.pkl', 'wb')
    pickle.dump(dataset, sentiment_dataset_pkl)
    sentiment_dataset_pkl.close()

    """



    sentences = [
        u'This is stupid ugly test.',
        u'This is a wonderful lovely test',
    ]
    predicted = gs_sclf.predict(sentences)
    confidence = gs_sclf.predict_proba(sentences)

    for s, p, con in zip(sentences, predicted, confidence):
        print(u'The polarity of "%s" is "%s", with probability "%0.2f"' % (s, dataset.target_names[p], con[p]))

    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    """