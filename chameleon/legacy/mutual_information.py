import numpy as np
import os
import jpype as jp
from scipy.stats import t
import pathlib


legacy_dir = pathlib.Path(__file__).parent.absolute()
jarLocation = legacy_dir / "infodynamics.jar" 

"""Global property defaults"""
# Calculator properties
mi_class_types = {
    'kraskov': 'infodynamics.measures.mixed.kraskov.MutualInfoCalculatorMultiVariateWithDiscreteKraskov',
    'kraskov_cont': 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov'
}
cond_mi_class_types = {
    'kraskov': 'infodynamics.measures.mixed.kraskov.ConditionalMutualInfoCalculatorMultiVariateWithDiscreteKraskov',
    'kraskov_cont': 'infodynamics.measures.continuous.kraskov.ConditionalMutualInfoCalculatorMultiVariateKraskov'
}
mi_calc_properties = {
    'normalise': 'false',
    'k': '4',
    'norm_type': 'max_norm'
}

# Kernel properties
num_samples = 1000
p_value_threshold = 0.05
max_joint_vars_in_iterative = 5
kNNsk = [1, 3, 5, 7, 9, 11, 13]

def binarise_y(y):
    y_vals = np.unique(y)
    for i in range(y.shape[0]):
        y[i] = 0 if y[i] == y_vals[0] else 1
    return y

def iterative_MI(X, y, **kwargs):
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + str(jarLocation))

    # Set properties from kwargs
    conditional = kwargs.get('conditional', 'true')
    max_features = kwargs.get('max_features', 50)
    continue_redundant = kwargs.get('continue_redundant', 'true')
    mi_calc_type = kwargs.get('mi_calc_type', 'kraskov')
    mi_class = mi_class_types[mi_calc_type]
    cond_mi_class = cond_mi_class_types[mi_calc_type]
    
    y = binarise_y(y)
    
    # Initialise calculator
    mi_calc_class = jp.JClass(mi_class)
    cond_mi_calc_class = jp.JClass(cond_mi_class)
    mi_calc = mi_calc_class()
    cond_mi_calc = cond_mi_calc_class()
    mi_calc_properties['normalise'] = kwargs.get('normalise', 'true')
    for prop, value in mi_calc_properties.items():
        mi_calc.setProperty(jp.JString(prop), jp.JString(value))
        cond_mi_calc.setProperty(prop, value)

    # Set up calculation variables 
    selected_features = []
    conditional_features = []
    insignificant_features = []
    unconditioned_MIs = np.array([])
    MIs = []
    surrogate_MIs = []
    surrogate_MI_stds = []
    candidates_to_add = np.arange(0, (X.shape[1]-len(selected_features)))
    last_feat_added_success = True
    reached_max_conditional_features = False
    conditional_features_changed = False
    redundancy_idx = -1
    j_y = jp.JArray(jp.JInt, 1)(y)
    # Perform feature selection... feature added in each iteration
    while len(selected_features) < max_features:
        conditional_data = X[:, conditional_features]
        if len(conditional_features) == 0 and len(unconditioned_MIs) != 0:
            candidate_MIs = unconditioned_MIs[candidates_to_add]
        else:
            candidate_MIs = np.array([0] * len(candidates_to_add), dtype=float)
            for cand_idx in range(0,len(candidates_to_add)):
                candidate = candidates_to_add[cand_idx]
                candidate_data = X[:, candidate]
                # print('MI for virtual sens number %d cond on: ', candidate)
                if(len(conditional_features) > 0):
                    try:
                        if len(conditional_features) == 1:
                            cond_mi_calc.initialise(1, 2, 1)
                            j_cand_data = jp.JArray(jp.JDouble, 1)(candidate_data)
                            j_cond_data = jp.JArray(jp.JDouble, 1)(conditional_data)
                            
                        else:
                            cond_mi_calc.initialise(2, 2, len(conditional_features))
                            list_cond = conditional_data.tolist()
                            j_cond_data = jp.JArray(jp.JDouble, conditional_data.shape[1])(list_cond)
                            candidate_data_j = np.zeros((y.shape[0], 2))
                            candidate_data_j[:, 0] = X[:, best_canditate_idx]
                            list_cand = candidate_data_j.tolist()
                            j_cand_data = jp.JArray(jp.JDouble, 2)(list_cand)
                            
                        cond_mi_calc.setObservations(j_cand_data, j_y, j_cond_data)
                        mi = cond_mi_calc.computeAverageLocalOfObservations()
                    except Exception as e: 

                        mi = -100            
                else:
                    j_cand_data = jp.JArray(jp.JDouble, 1)(candidate_data)
                    j_y = jp.JArray(jp.JInt, 1)(y)
                    mi_calc.initialise(1, 2)
                    mi_calc.setObservations(j_cand_data, j_y)
                    mi = mi_calc.computeAverageLocalOfObservations()
                candidate_MIs[cand_idx] = mi
                conditional_features_changed = False
            if len(unconditioned_MIs) == 0:
                unconditioned_MIs = candidate_MIs
        sorted_indices = np.argsort(candidate_MIs)[::-1]
        sorted_MIs = np.sort(candidate_MIs)[::-1]
        best_MI = sorted_MIs[0]

        best_canditate_idx = candidates_to_add[sorted_indices[0]]
        print(f'Best virtual sensor to add is {best_canditate_idx}, giving MI {best_MI} bits/nats')
        # Check the statistical significance of the conditional MI value:
        candidate_data = X[:,best_canditate_idx]
        if(len(conditional_features) > 0):
            cond_mi_calc.initialise(len(conditional_features), 2, len(conditional_features))
            mi = 0
            try:
                if len(conditional_features) == 1:
                    j_cand_data = jp.JArray(jp.JDouble, 1)(candidate_data)
                    j_cond_data = jp.JArray(jp.JDouble, 1)(conditional_data)
                else:
                    
                    
                    candidate_data_j = np.zeros((y.shape[0], y.shape[0]))
                    candidate_data_j[:, 0] = X[:, best_canditate_idx]
                    list_cand = candidate_data_j.tolist()
                    j_cand_data = jp.JArray(jp.JDouble, 2)(list_cand)
                    list_cond = conditional_data.tolist()
                    j_cond_data = jp.JArray(jp.JDouble, len(conditional_features))(list_cond)
                cond_mi_calc.setObservations(j_cand_data, j_y, j_cond_data)
                mi = cond_mi_calc.computeAverageLocalOfObservations()
                m_dist = cond_mi_calc.computeSignificance(False, num_samples)
            except:
                mi = 0
                print("Java CholeskyDecomposition Exception during MI calculation or stat significance check")
                list_zeros = np.concatenate((np.zeros(num_samples),np.ones(num_samples)))
                j_j = jp.JArray(jp.JDouble, 1)(list_zeros)
                m_dist_dummy_class = jp.JClass('infodynamics.utils.EmpiricalMeasurementDistribution')
                m_dist = m_dist_dummy_class(j_j, mi )
        else:
            mi_calc.initialise(1, 2)        
            j_cand_data = jp.JArray(jp.JDouble, 1)(candidate_data)
            mi_calc.setObservations(j_cand_data, j_y)
            mi = mi_calc.computeAverageLocalOfObservations()
            m_dist = mi_calc.computeSignificance(num_samples)
        surrogate_MI = m_dist.getMeanOfDistribution()
        surrogate_MI_std = m_dist.getStdOfDistribution()
        t_value = (mi - surrogate_MI) / surrogate_MI_std
        signif_level = t.cdf(t_value, num_samples)
        if signif_level > 1 - p_value_threshold:
            passed_stats_test = True
            print(' --> accepted at p=%f' % signif_level)
        else:
            print(' --> rejected at p=%f' % signif_level)
            passed_stats_test = False
        accepting_feature = False
        if passed_stats_test:
            accepting_feature = True
            last_feat_added_success = True
        else:
            if redundancy_idx == -1:
                redundancy_idx = len(selected_features)
            if continue_redundant:
                if len(conditional_features) > 0:
                    print('Clearing set of variables to condition on, and trying to select redundant variables')
                    conditional_features = []
                    conditional_features_changed = True
                    print(len(conditional_features))
                    last_feat_added_success = False
                else:
                    print('Accepting variable anyway, since we''re forcing redundant selection and are not conditioning on any variables at this round')
                    accepting_feature = True
                    insignificant_features.append(best_canditate_idx)
                    last_feat_added_success = True
            else:
                last_feat_added_success = False
                print('Terminating forward feature selection as we have reached maximum prescribed number of sensors: %d of %d' % (len(selected_features), max_features))                
                break
        
        if accepting_feature:
            print('Accepted as %dth virtual sensor' % (len(selected_features) + 1))
            ####################################
            selected_features.append(best_canditate_idx)
            if conditional:
                conditional_features.append(best_canditate_idx)
                conditional_features_changed = True
                print(len(conditional_features))
            if reached_max_conditional_features:
                conditional_features = []
                conditional_features_changed = True
                print('Clearing set of variables to condition on, since we''ve reached %d of them' % max_features)
            ####################################
            MIs.append(mi)
            surrogate_MIs.append(surrogate_MI)
            surrogate_MI_stds.append(surrogate_MI_std)
            print('removing best from canddiates to add')
            candidates_to_add = candidates_to_add[candidates_to_add != best_canditate_idx ]
        ####################################
    # return the results
    results_dict = {
        'selected_features': selected_features,
        'mi': mi,
        'surrogate_MI': surrogate_MI,
        'surrogate_MI_std': surrogate_MI_std,
        'last_feat_added_success': last_feat_added_success,
        'insignificant_features': insignificant_features,
        'MIs': MIs,
        'surrogate_MIs': surrogate_MIs,
        'surrogate_MI_stds': surrogate_MI_stds,
        'redundancy_idx': redundancy_idx
    } 

    return results_dict

def simple_MI(X, y, **kwargs):
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + str(jarLocation))

    # Set properties from kwargs
    mi_calc_type = kwargs.get('mi_calc_type', 'kraskov')
    mi_class = mi_class_types[mi_calc_type]
    cond_mi_class = cond_mi_class_types[mi_calc_type]
    # Initialise calculator
    mi_calc_class = jp.JClass(mi_class)
    cond_mi_calc_class = jp.JClass(cond_mi_class)
    mi_calc = mi_calc_class()
    cond_mi_calc = cond_mi_calc_class()
    for prop, value in mi_calc_properties.items():
        mi_calc.setProperty(jp.JString(prop), jp.JString(value))
        cond_mi_calc.setProperty(prop, value)

    y = binarise_y(y)

    # Set up calculation variables 
    MIs = []
    
    # Perform feature selection... feature added in each iteration
    for cand_idx in range(0,X.shape[1]):
        mi_calc.initialise(1, 2)
        X_j = jp.JArray(jp.JDouble, 1)(X[:,cand_idx])
        y_j = jp.JArray(jp.JInt, 1)(y)
        mi_calc.setObservations(X_j, y_j)
        mi = mi_calc.computeAverageLocalOfObservations()
        
        MIs.append(mi)
        
    sorted_indices = np.argsort(MIs)[::-1]
    sorted_MIs = np.sort(MIs)[::-1]

    return sorted_indices
    
