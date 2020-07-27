from sys import  argv
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE


import chameleon.write_files as write
import chameleon.read_files as read
from chameleon.legacy.skfeature.function.similarity_based import fisher_score, reliefF

# from chameleon.legacy.mutual_information import iterative_greedy, simple_MI



def apply_fischer(X, y):
    score = fisher_score.fisher_score(X, y)
    return fisher_score.feature_ranking(score)


def apply_reliefF(X, y, **kwargs):
    k = kwargs.get('k', 5)
    score = reliefF.reliefF(X, y, k=k)
    return reliefF.feature_ranking(score)


def apply_SVM_RFE(X, y, **kwargs):
    n_features = kwargs.get('n_features', 1)
    step = kwargs.get('step', 1) 
    feature_subset = np.arange(X.shape[1])
    feature_idx_elimination_order = []
    for i in range(n_features, 0, -1):
        X_set = X[:, feature_subset]
        svc = LinearSVC()
        rfe = RFE(svc, i, step=step, verbose=1)
        rfe.fit(X_set, y)
        boolean_mask = rfe.get_support(indices=False)
        pruned_feature_indices = feature_subset[np.invert(boolean_mask)]
        feature_idx_elimination_order.extend(list(pruned_feature_indices))
        feature_subset = feature_subset[boolean_mask]

    # Add the unpruned features 
    feature_idx_elimination_order.extend(feature_subset)
    return feature_idx_elimination_order[::-1], rfe


def apply_RF(X, y, **kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X, y)
    selected_features = np.argsort(rf.feature_importances_)[::-1]
    return selected_features


# def apply_iterative_MI(dataset, fold_idx, version, **kwargs):
#     results = iterative_greedy(
#         dataset,    
#         fold_idx, 
#         version, 
#         **kwargs)
#     return results


# def apply_simple_MI(dataset, fold_idx, version, **kwargs):
#     selected_features = simple_MI(dataset, fold_idx, version)
#     return selected_features



def run(datafile, algorithm):
    df = read.read_pandas(datafile)
    X = df['X_train']
    y = df['y_train'].flatten()

    if algorithm == 'fischer':
        feature_idx = apply_fischer(X, y)
    elif algorithm == 'reliefF':
        n_neighbours = 5
        feature_idx = apply_reliefF(X, y, k=n_neighbours)
    elif algorithm == 'SVM-RFE':
        step = 1
        n_features = 50
        feature_idx, rfe = apply_SVM_RFE(X, y, n_features=n_features, step=step)
    elif algorithm == 'random-forest':
        n_estimators = 100
        feature_idx = apply_RF(X, y, n_estimators=n_estimators)
    # elif algorithm == 'iterative_MI':
    #     results = apply_iterative_MI(
    #         dataset,    
    #         fold_idx, 
    #         version, 
    #         X=X, 
    #         y=y, 
    #         max_features=n_features)
    #     feature_idx = results['selected_features']
    # elif algorithm == 'simple_MI':
    #     feature_idx = apply_simple_MI(
    #         dataset,    
    #         fold_idx, 
    #         version, 
    #         X=X, 
    #         y=y)

    # Log runtime
    # runtime = time.time() - start_time
    # file_access.log_runtime(runtime, dataset, fold_idx, version, selector)

    # Save feature selection output
    # save_features(feature_idx, dataset, fold_idx, version, selector)
    return feature_idx