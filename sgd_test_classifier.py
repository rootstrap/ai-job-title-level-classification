import pickle
from test_functions import load_instances, test_classifier


# Load instances
clf, count_vect, tfidf_transformer = load_instances('sgd')
X_test = pickle.load(open('data_process/data_sets/x_test.pkl', 'rb'))

# Test and show results
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
test_predict = clf.predict(X_test_tfidf)
test_classifier(clf, test_predict)
