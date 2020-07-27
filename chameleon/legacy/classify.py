import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC  
from sklearn.metrics import accuracy_score

import chameleon.read_files as read



classifers = {
    'naive-bayes': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'logistic-regression': LogisticRegression(),
    'neural-net': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,100)),
    'random-forest': RandomForestClassifier(),
    'SVM': LinearSVC()
}

def predict(clf, X_train, X_test, y_train, y_test):
    model = classifers[clf]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def run(datapath, clf, feature_idx, n_features):
    df = read.read_pandas(datapath)
    
    predictions = predict(clf, 
                        df['X_train'][:,feature_idx[:n_features]], 
                        df['X_test'][:,feature_idx[:n_features]],
                        df['y_train'].flatten(),
                        df['y_test'].flatten()
                        )
    
    accuracy = accuracy_score(df['y_test'].flatten(), predictions)
    return accuracy
