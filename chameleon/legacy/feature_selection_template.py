from sys import  argv
import time

import chameleon.write_files as write
import chameleon.read_files as read

# import feature_selection as fs
# from file_access import load_fold_set, load_raw_data, save_features, save_model
# from data_manipulation import apply_fold
# import file_access

# Set options

def run_feature_selection(datafile, algorithm):
    X, y = read.read_pandas(datafile)
    print('hello')


# fold_set = load_fold_set(dataset, version)
# fold = fold_set[int(fold_idx)]
# skip_normalise = selector in ['iterative_MI']
# data = apply_fold(dataset, fold, X_raw, y_raw, skip_normalise=skip_normalise)
# X = data['X_train']
# y = data['y_train']

# n_features = 50
# feature_idx = []
# start_time = time.time()

# if selector == 'skfeature_fischer':
#     feature_idx = fs.apply_fischer(X, y)
# elif selector == 'skfeature_gini':
#     feature_idx = fs.apply_gini(X, y)
# elif selector == 'skfeature_reliefF':
#     n_neighbours = 5
#     feature_idx = fs.apply_reliefF(X, y, k=n_neighbours)
# elif selector == 'skfeature_MIM':
#     feature_idx, JCMI, MIy = fs.apply_MIM(X, y, n_features=n_features)
# elif selector == 'sklearn_SVM-RFE':
#     step = 1
#     feature_idx, rfe = fs.apply_SVM_RFE(X, y, n_features=n_features, step=step)
# elif selector == 'sklearn_variance':
#     threshold = 0
#     feature_idx = fs.apply_var(X, threshold=threshold)
# elif selector == 'skfeature_FCBF':
#     delta = 0
#     feature_idx, sel_feature_SU = fs.apply_FCBF(X, y, delta=delta)
# elif selector == 'sklearn_random-forest':
#     n_estimators = 100
#     feature_idx = fs.apply_RF(X, y, n_estimators=n_estimators)
# elif selector == 'iterative_MI':
#     results = fs.apply_iterative_MI(
#         dataset,    
#         fold_idx, 
#         version, 
#         X=X, 
#         y=y, 
#         max_features=n_features)
#     save_model(results, dataset, fold_idx, version, selector)
#     feature_idx = results['selected_features']
# elif selector == 'simple_MI':
#     feature_idx = fs.apply_simple_MI(
#         dataset,    
#         fold_idx, 
#         version, 
#         X=X, 
#         y=y)

# # Log runtime
# runtime = time.time() - start_time
# file_access.log_runtime(runtime, dataset, fold_idx, version, selector)

# # Save feature selection output
# save_features(feature_idx, dataset, fold_idx, version, selector)

