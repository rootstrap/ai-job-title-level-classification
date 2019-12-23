
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from fit_tune_function import fit_tune_store_sgdcv

X_train = pickle.load(open('data_process/data_sets/x_train.pkl', 'rb'))
len_n = len(X_train)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)),
])

parameters = {
    'vect__ngram_range': [(1, 1), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (1, 21, 33, 42, 88, 160, ),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-5, ),
    'clf__hidden_layer_sizes': [(1, ), (3, ), (len_n, ), (int(len_n/2), int(len_n/2)), ]
}

mlp_clf_gscv = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
fit_tune_store_sgdcv(mlp_clf_gscv, 'mlp')
