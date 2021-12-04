from threading import local
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from growingspheres import counterfactuals as cf
from growingspheres.utils.gs_utils import generate_inside_field, generate_categoric_inside_ball
import random
from sklearn.metrics import precision_score
from growingspheres.utils.gs_utils import distances

def find_closest_counterfactual(ape, instance, growing_method):
    ape.target_class = ape.black_box_predict(instance.reshape(1, -1))[0]
    # Computes the distance to the farthest instance from the training dataset to bound generating instances 
    farthest_distance = 0
    for training_instance in ape.train_data:
        farthest_distance_now = distances(training_instance, instance, ape)
        if farthest_distance_now > farthest_distance:
            farthest_distance = farthest_distance_now
    growing_sphere = cf.CounterfactualExplanation(instance, ape.black_box_predict, method=growing_method, target_class=None, 
                                                continuous_features=ape.continuous_features, 
                                                categorical_features=ape.categorical_features, 
                                                categorical_values=ape.categorical_values,
                                                max_features=ape.max_features,
                                                min_features=ape.min_features)

    growing_sphere.fit(n_in_layer=2000, first_radius=0.001, dicrease_radius=10, sparse=True, 
                                                verbose=ape.verbose, feature_variance=ape.feature_variance, 
                                                farthest_distance_training_dataset=farthest_distance, 
                                                probability_categorical_feature=ape.probability_categorical_feature, 
                                                min_counterfactual_in_sphere=ape.nb_min_instance_per_class_in_sphere)
    closest_counterfactual = growing_sphere.enemy
    min_instance_per_class = ape.nb_min_instance_per_class_in_sphere
    position_training_instances_in_sphere, nb_training_instance_in_sphere = ape.instances_from_dataset_inside_sphere(instance, 
                                                                                                growing_sphere.radius, ape.train_data)
    training_instances_in_sphere, train_labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = \
                                                                                ape.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                instance, ape.train_data, farthest_distance, 
                                                                                                min_instance_per_class, position_training_instances_in_sphere, 
                                                                                                nb_training_instance_in_sphere, libfolding=True,
                                                                                                growing_method=growing_method)
    position_testing_instances_in_sphere, nb_testing_instance_in_sphere = ape.instances_from_dataset_inside_sphere(instance, 
                                                                                                growing_sphere.radius, ape.test_data)
    test_instances_in_sphere, test_labels_in_sphere, test_percentage_distribution, _ = ape.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                instance, ape.test_data,
                                                                                                farthest_distance, 
                                                                                                ape.nb_min_instance_per_class_in_sphere,
                                                                                                position_testing_instances_in_sphere, 
                                                                                                nb_testing_instance_in_sphere,
                                                                                                growing_method=growing_method)
    return closest_counterfactual, growing_sphere, training_instances_in_sphere, train_labels_in_sphere, test_instances_in_sphere,\
            test_labels_in_sphere, instances_in_sphere_libfolding, farthest_distance, position_training_instances_in_sphere, nb_training_instance_in_sphere

def user_experiments_gs_gf(instance, ape, nb_features_employed):
    target_class = ape.black_box_predict(instance.reshape(1, -1))[0]
    growing_methods = ['GS', 'GF']
    features_employed = []
    for growing_method in growing_methods:
        closest_counterfactual, growing_sphere, training_instances_in_sphere, train_labels_in_sphere, test_instances_in_sphere, \
                                                    test_labels_in_sphere, instances_in_sphere_libfolding, _, \
                                                    _, _\
                                                    =find_closest_counterfactual(ape, instance, growing_method)
        local_surrogate_exp = ape.lime_explainer.explain_instance_training_dataset(closest_counterfactual, 
                                                                                ape.black_box_predict_proba, 
                                                                                num_features=nb_features_employed, 
                                                                                instances_in_sphere=training_instances_in_sphere,
                                                                                ape=ape)
        features_local_surrogate_employed = []
        for feature_local_surrogate_employed in local_surrogate_exp.as_list():
            features_local_surrogate_employed.append(feature_local_surrogate_employed[0])
        rules, training_instances_pandas_frame, features_employed_in_local_surrogate = ape.generate_rule_and_data_for_anchors(features_local_surrogate_employed, 
                                                                                                    target_class, ape.train_data, 
                                                                                                    simulated_user_experiment=True)

        features_employed_in_local_surrogate.sort()
        features_employed_in_local_surrogate = list(set(features_employed_in_local_surrogate))
        anchor_exp = ape.anchor_explainer.explain_instance(instance, ape.black_box_predict, threshold=ape.threshold_precision, 
                                delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, stop_on_first=False,
                                desired_label=None, beam_size=4)
        print("rule by anchor", anchor_exp.names())
        #print("rule by anchor", anchor_exp.names()) => To print the rules returned by Anchors
        # Transform the explanation generated by Anchors to know what are the features employed by Anchors
        rules, training_instances_pandas_frame, features_employed_in_rule = ape.generate_rule_and_data_for_anchors(anchor_exp.names(), 
                                                                                                ape.target_class, ape.train_data, 
                                                                                                simulated_user_experiment=True)
        features_employed_in_rule = list(set(features_employed_in_rule))
        features_employed.append(features_employed_in_local_surrogate)
        features_employed.append(features_employed_in_rule)
    return features_employed[0], features_employed[2], features_employed[1], features_employed[3]

def ape_center(ape, instance, growing_method='GF', nb_features_employed=None):
    closest_counterfactual, growing_sphere, training_instances_in_sphere, train_labels_in_sphere, test_instances_in_sphere, \
                            test_labels_in_sphere, instances_in_sphere_libfolding, farthest_distance, \
                            position_training_instances_in_sphere, nb_training_instance_in_sphere = \
                                                    find_closest_counterfactual(ape, instance, growing_method)
    """ Generates or store instances in the area of the hyperfield and their corresponding labels """
    
    if instances_in_sphere_libfolding != []:
        # In case of categorical data, we transform categorical values into probability distribution (continuous values for libfolding)
        index_counterfactual_instances_in_sphere = ape.store_counterfactual_instances_in_sphere(training_instances_in_sphere, libfolding=True)
        counterfactual_instances_in_sphere = training_instances_in_sphere[index_counterfactual_instances_in_sphere]
        counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
        #print("start of unimodality test")
        unimodal_test = ape.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius,
                                                        counterfactual_libfolding=counterfactual_libfolding)
    else:
        counterfactual_instances_in_sphere = ape.store_counterfactual_instances_in_sphere(training_instances_in_sphere)
        unimodal_test = ape.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius)
    nb = 0
    min_instance_per_class = ape.nb_min_instance_per_class_in_sphere
    while not unimodal_test:
        # While the libfolding test is not able to declare that data are multimodal or unimodal we extend the number of instances that are generated
        min_instance_per_class *= 1.5
        training_instances_in_sphere, train_labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = \
                                    ape.generate_instances_inside_sphere(growing_sphere.radius, instance, ape.train_data, farthest_distance, 
                                                                                                min_instance_per_class, position_training_instances_in_sphere, 
                                                                                                nb_training_instance_in_sphere, libfolding=True,
                                                                                                growing_method=growing_method)
        
        if instances_in_sphere_libfolding != []:
            index_counterfactual_instances_in_sphere = ape.store_counterfactual_instances_in_sphere(training_instances_in_sphere, libfolding=True)
            counterfactual_instances_in_sphere = training_instances_in_sphere[index_counterfactual_instances_in_sphere]
            counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
            unimodal_test = ape.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), 
                                                            training_instances_in_sphere, growing_sphere.radius,
                                                            counterfactual_libfolding=counterfactual_libfolding)
        else:
            counterfactual_instances_in_sphere = ape.store_counterfactual_instances_in_sphere(training_instances_in_sphere)
            unimodal_test = ape.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius)
        if ape.verbose:
            print("nb times libfolding is not able to determine wheter datas are unimodal or multimodal:", nb)
            print("There are ", len(counterfactual_instances_in_sphere), " instances in the datas given to libfolding.")
            print()
        nb += 1
    """ Computes the labels for instances from the dataset to compute precision for explanation method """
    labels_instance_test_data = ape.black_box_predict(ape.test_data)
    nb_instance_test_data_label_as_target = sum(x == ape.target_class for x in labels_instance_test_data)
    
    
    anchor_precision, anchor_coverage, anchor_f2, anchor_explanation = ape.compute_anchor_precision_coverage(instance, 
                                    labels_instance_test_data, len(test_instances_in_sphere), 
                                    farthest_distance, percentage_distribution, nb_instance_test_data_label_as_target,
                                    growing_method=growing_method)
    ls_precision, ls_coverage, ls_f2, ls_explanation, ape.extended_radius = ape.compute_lime_extending_precision_coverage(training_instances_in_sphere, 
                                    instance, train_labels_in_sphere, growing_sphere, nb_features_employed,
                                    farthest_distance, growing_method)
    if ape.multimodal_results:
        # In case of multimodal data, we generate a rule based explanation and compute precision and coverage of this explanation model
        ape_precision, ape_coverage, ape_f2, ape_explanation = anchor_precision, anchor_coverage, anchor_f2, anchor_explanation

    else:
        # In case of unimodal data, we generate linear explanation and compute precision and coverage of this explanation model
        ape_precision, ape_coverage, ape_f2, ape_explanation = ls_precision, ls_coverage, ls_f2, ls_explanation
    return [ls_precision, anchor_precision, ape_precision], [ls_coverage, anchor_coverage, ape_coverage], [ls_f2, anchor_f2, ape_f2], \
                ape.multimodal_results, ape.extended_radius, ape.separability_index, ape.pvalue 

def compute_other_linear_explanation_precision_coverage(ape_tabular, nb_testing_instance_in_sphere, position_instances_in_sphere,
                                                            instances_in_sphere, labels_in_sphere, nb_instance_test_data_label_as_target, 
                                                            nb_testing_instance_in_sphere_label_as_target, labels_testing_instance_in_sphere, 
                                                            nb_features_employed, growing_spheres, growing_method):
    """
    Computation of precision and coverage to compare multiple local surrogate explanation model
    Args: ape_tabular: Ape tabular object used to explain instance
            nb_testing_instance_in_sphere: Number of instances from the training data that are located in the hyper field
            position_instances_in_sphere: index of the intances from the training data that are located in the hyper field
            instances_in_sphere: Artificial and training data located in the hyper field
            labels_in_sphere: Black box model classification of instances in the hyper field
            nb_instance_train_data_label_as_target: Number of instances from the training data that are classify as the target instance
            nb_testing_instance_in_sphere_label_as_target: Number of instances from the training data located in the hyper field classify as the target instance
            labels_training_instance_in_sphere: Labels of training instances located in the hyper field
            nb_features_employed: Number of features employed as explanation by local surrogate
    Return: Precision, Coverage and F1 for classic local surrogate, local surrogate train over training data with a linear regression 
            and a local surrogate using a logistic regression model as explanation
    """
    # Trained a local surrogate model over binarize data using a linear regression (default Local Surrogate)
    local_surrogate = ape_tabular.lime_explainer.explain_instance(ape_tabular.closest_counterfactual, 
                                                                ape_tabular.black_box_predict_proba, num_features=nb_features_employed)
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(local_surrogate, instances_in_sphere)
    precision_local_surrogate = ape_tabular.compute_linear_regression_precision(prediction_inside_sphere, labels_in_sphere)

    """
    # Trained a local surrogate model over training data using a linear regression (by default on Lime)
    local_surogate_linear_regression = ape_tabular.lime_explainer.explain_instance_training_dataset(ape_tabular.closest_counterfactual, 
                                                                ape_tabular.black_box_predict_proba, num_features=nb_features_employed,
                                                                model_regressor=LogisticRegression())
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(local_surogate_linear_regression, instances_in_sphere)
    precision_local_surogate_linear_regression = ape_tabular.compute_linear_regression_precision(prediction_inside_sphere, labels_in_sphere)
    """
    # Trained a local surrogate model over training data using a linear regression (by default on Lime)
    local_surogate_raw_data = ape_tabular.lime_explainer.explain_instance_training_dataset(ape_tabular.closest_counterfactual, 
                                                                ape_tabular.black_box_predict_proba, num_features=nb_features_employed,
                                                                instances_in_sphere=instances_in_sphere,
                                                                ape=ape_tabular)
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(local_surogate_raw_data, instances_in_sphere)
    precision_local_surogate_raw_data = ape_tabular.compute_linear_regression_precision(prediction_inside_sphere, labels_in_sphere)
    
    # Compute precision for a local Surrogate model with a Logistic Regression model as explanation model over binary data
    local_surrogate_exp_regression = ape_tabular.lime_explainer.explain_instance(ape_tabular.closest_counterfactual,  
                                                            ape_tabular.black_box_predict, num_features=nb_features_employed,
                                                            model_regressor = LogisticRegression())
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(local_surrogate_exp_regression, instances_in_sphere)
    precision_local_surrogate_logistic_regression = max(precision_score(labels_in_sphere, prediction_inside_sphere, pos_label=ape_tabular.target_class),
                                                        precision_score(labels_in_sphere, prediction_inside_sphere, pos_label=1-ape_tabular.target_class))
    
    precision_ls_raw_data, lime_extending_coverage, f2_lime_extending, ls_explanation, radius = ape_tabular.compute_lime_extending_precision_coverage(instances_in_sphere,
                                                ape_tabular.closest_counterfactual, labels_in_sphere, growing_spheres, 
                                                nb_features_employed, 1, growing_method)

    if ape_tabular.verbose: print("Computing multiple linear explanation models precision and coverage.")
    linear_coverage = nb_testing_instance_in_sphere_label_as_target/nb_instance_test_data_label_as_target
    f2_lime_regression = (2*precision_local_surrogate_logistic_regression+linear_coverage)/3
    f2_not_bin_lime = (2*precision_local_surogate_raw_data+linear_coverage)/3
    f2_local_surrogate = (2*precision_local_surrogate+linear_coverage)/3
    return [precision_local_surrogate, precision_local_surrogate_logistic_regression, precision_local_surogate_raw_data, precision_ls_raw_data], \
                [linear_coverage, linear_coverage, linear_coverage, lime_extending_coverage], \
                    [f2_local_surrogate, f2_lime_regression, f2_not_bin_lime, f2_lime_extending]

def compute_all_explanation_method_precision(ape_tabular, instance, growing_sphere, dicrease_radius, radius,
                                                nb_training_instance_in_sphere, nb_instance_test_data_label_as_target,
                                                position_instances_in_sphere, instances_in_sphere, labels_in_sphere,
                                                farthest_distance,  farthest_distance_cf, percentage_distribution, nb_features_employed,
                                                growing_method):
    """
    Compute Precision, Coverage and F1 for APE, Anchors and the best Local Surrogate explanation models
    Args: ape_tabular: Ape tabular object used to explain the target instance
            instance: Target instance to explain
            growing sphere: Growing Sphere object used by APE
            dicrease_radius: Ratio for dicreasing the radius of the hyper field
            radius: Radius of the hyper field
            nb_training_instance_in_sphere: number of training data located in the hyper field
            nb_instance_train_data_label_as_target: Number of instances from the training data that are label as the target instance
            position_instances_in_sphere: Index of the instances from the training data located in the hyper field
            instances_in_sphere: Set of instances (artificial and training data) located in the hyper field
            labels_in_sphere: Classification of instances in the hyper field by the black box model
            farthest_distance: Distance from the target instance to the farthest instance from the training data
            percentage_distribution: Percentage of categorical values that are changing during growing field algorithm 
                                    (radius of the field for categorical data)
            nb_features_employed: Number of features employed by the linear explanation model
    Return: Arrays of precision, coverage and F1 of multiple explanation models
    """
    labels_instance_test_data = ape_tabular.black_box_predict(ape_tabular.test_data)
    ape_tabular.instance_to_explain = instance
    
    # Compute precision, coverage and F2 of local surrogate trained over raw data, with an extending sphere and a logistic regression model as explanation
    local_surrogate_extend_raw_precision, local_surrogate_extend_raw_precision_log, local_surrogate_extend_raw_coverage, f2_local_surrogate_extend_raw, _, \
                        ape_tabular.extended_radius = ape_tabular.compute_lime_extending_precision_coverage(instances_in_sphere, 
                                            ape_tabular.closest_counterfactual, labels_in_sphere, growing_sphere, nb_features_employed, 
                                            farthest_distance_cf, growing_method, position_instances_in_sphere)

    # Compute precision, coverage and F2 for Anchors
    anchor_precision, anchor_coverage, f2_anchor, _ = ape_tabular.compute_anchor_precision_coverage(instance, 
                                    labels_instance_test_data, len(instances_in_sphere), 
                                    farthest_distance, percentage_distribution, nb_instance_test_data_label_as_target,
                                    growing_method)
    
    # Compute precision, coverage and F2 for Decision Tree
    decision_tree_precision, decision_tree_coverage, f2_decision_tree, _, \
                decision_tree_radius = ape_tabular.compute_decision_tree_precision_coverage(instances_in_sphere, labels_in_sphere, 
                                        ape_tabular.closest_counterfactual, growing_sphere.radius, position_instances_in_sphere)

    # Compute precision for classic Local Surrogate
    local_surrogate = ape_tabular.lime_explainer.explain_instance(ape_tabular.closest_counterfactual,
                                                                ape_tabular.black_box_predict_proba, 
                                                                #ape_tabular.black_box_predict, model_regressor = LogisticRegression(),
                                                                num_features=nb_features_employed)
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(local_surrogate, instances_in_sphere)
    precision_local_surrogate = {'real':None}
    if position_instances_in_sphere != []:
        real_prediction_in_fields = ape_tabular.modify_instance_for_linear_model(local_surrogate, ape_tabular.train_data[position_instances_in_sphere])
        real_labels_in_sphere = ape_tabular.black_box_predict(ape_tabular.train_data[position_instances_in_sphere])
        precision_local_surrogate["real"] = ape_tabular.compute_linear_regression_precision(real_prediction_in_fields, real_labels_in_sphere)
        #precision_score(real_labels_in_sphere, real_prediction_in_fields, pos_label=ape_tabular.target_class) 
    precision_local_surrogate['all'] = ape_tabular.compute_linear_regression_precision(prediction_inside_sphere, labels_in_sphere)
    #precision_score(labels_in_sphere, prediction_inside_sphere, pos_label=ape_tabular.target_class)

    # Compute coverage and F2 for classic Local Surrogate
    position_testing_instances_in_sphere, nb_testing_instance_in_sphere = ape_tabular.instances_from_dataset_inside_sphere(ape_tabular.closest_counterfactual, 
                                                                                                                growing_sphere.radius, ape_tabular.test_data)
    nb_testing_instance_in_sphere_label_as_target, labels_testing_instance_in_sphere = ape_tabular.compute_labels_inside_sphere(nb_testing_instance_in_sphere, 
                                                                                                                                position_testing_instances_in_sphere,
                                                                                                                                ape_tabular.test_data)
    linear_coverage = nb_testing_instance_in_sphere_label_as_target/nb_instance_test_data_label_as_target
    f2_linear_surrogate = (2%precision_local_surrogate['all'] + linear_coverage)/3

    # Select values for APE depending on the unimodality test
    ape_precision = anchor_precision if ape_tabular.multimodal_results else local_surrogate_extend_raw_precision
    ape_coverage = anchor_coverage if ape_tabular.multimodal_results else local_surrogate_extend_raw_coverage
    f2_ape = f2_anchor if ape_tabular.multimodal_results else f2_local_surrogate_extend_raw
    """
    ape_precision = decision_tree_precision if ape_tabular.multimodal_results else local_surrogate_extend_raw_precision
    ape_coverage = decision_tree_coverage if ape_tabular.multimodal_results else local_surrogate_extend_raw_coverage
    f2_ape = f2_decision_tree if ape_tabular.multimodal_results else f2_local_surrogate_extend_raw
    """

    precisions = [precision_local_surrogate['all'], local_surrogate_extend_raw_precision_log['all'], local_surrogate_extend_raw_precision['all'], \
                anchor_precision['all'], ape_precision['all'], decision_tree_precision['all']]
    coverages = [linear_coverage, local_surrogate_extend_raw_coverage, local_surrogate_extend_raw_coverage, anchor_coverage, ape_coverage, decision_tree_coverage]
    f2s = [f2_linear_surrogate, (local_surrogate_extend_raw_precision_log['all'] + local_surrogate_extend_raw_coverage)/2, f2_local_surrogate_extend_raw,  \
                f2_anchor, f2_ape, f2_decision_tree]
    multimodal = 1 if ape_tabular.multimodal_results else 0
    precisions_real = [precision_local_surrogate['real'], local_surrogate_extend_raw_precision['real'], local_surrogate_extend_raw_precision['real'], \
                anchor_precision['real'], ape_precision['real'], decision_tree_precision['real']]
    return precisions, coverages, f2s, multimodal, ape_tabular.extended_radius, precisions_real

def compute_local_surrogate_precision_coverage(ape_tabular, instance, growing_sphere,
                                                test_instances_in_sphere, test_labels_in_sphere,
                                                position_testing_instances_in_sphere, nb_testing_instance_in_sphere, 
                                                nb_features_employed, growing_method):
    """
    Compute precision, coverage and f2 for multiple linear explanation models
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: target instance to explain
          growing sphere: growing sphere object used by APE
          instances_in_sphere: (artificial and trained) data located in the hyper field
          labels_in_sphere: Labels by the black box model of the instances located in the hyper field
          position_instances_in_sphere: Index of the instaces from the training data located in the field
          nb_training_instance_in_sphere: Number of training data located in the hyper field
          nb_features_employed: Number of features employed by the linear explanations models
    Return: precision, coverage and f2 of classic local surrogate, a local surrogate train over training data with a linear regression 
            and a local surrogate using a logistic regression model as explanation
    """
    labels_instance_test_data = ape_tabular.black_box_predict(ape_tabular.test_data)
    nb_instance_test_data_label_as_target = sum(x == ape_tabular.target_class for x in labels_instance_test_data)
    
    nb_testing_instance_in_sphere_label_as_target, labels_testing_instance_in_sphere = ape_tabular.compute_labels_inside_sphere(nb_testing_instance_in_sphere, 
                                                                                                                                position_testing_instances_in_sphere,
                                                                                                                                ape_tabular.test_data)    
    
    precision, coverage, f2 = compute_other_linear_explanation_precision_coverage(ape_tabular,
                                                            nb_testing_instance_in_sphere, position_testing_instances_in_sphere, 
                                                            test_instances_in_sphere, test_labels_in_sphere, nb_instance_test_data_label_as_target, 
                                                            nb_testing_instance_in_sphere_label_as_target, 
                                                            labels_testing_instance_in_sphere, nb_features_employed,
                                                            growing_sphere, growing_method)
    return precision, coverage, f2



def simulate_user_experiments(ape_tabular, instance, nb_features_employed, farthest_distance, closest_counterfactual, growing_sphere,
                              position_instances_in_sphere, nb_training_instance_in_sphere, only_anchors=False, only_lse=False):
    """
    Function that generate a classic local surrogate explanation, an anchor explanation and an APE explanation and 
    returns the features that are used by these explanation models
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: target instance to explain
          nb_features_employed: Number of features employed by the linear explanation model
          farthest_distance: Distance from the target instance to explain and the farthest instance from the training data
          closest_counterfactual: Closest instance from the closest counterfactual from the target instance (on the limit of the hyper field 
                                  from the same class as the target instance)
          growing_sphere: growing sphere object used by APE to explain the target instance
          position_instances_in_sphere: Index of the training data located in the hyper field
          nb_training_instance_in_sphere: Number of training data located in the hyper field
    Return: List of features employed by APE, Anchors and a classic local surrogate
    """
    ape_tabular.target_class = ape_tabular.black_box_predict(instance.reshape(1, -1))[0]
    
    """ Generates or store instances in the area of the hypersphere and their correspoinding labels """
    min_instance_per_class = ape_tabular.nb_min_instance_per_class_in_sphere
    instances_in_sphere, labels_in_sphere, percentage_distribution, _ = ape_tabular.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                    closest_counterfactual, ape_tabular.train_data, farthest_distance, 
                                                                                                    min_instance_per_class, position_instances_in_sphere, 
                                                                                                    nb_training_instance_in_sphere, libfolding=False)
    if only_lse:
        #if not ape_tabular.multimodal_results:
            # In case of unimodal data we compute a local surrogate explanation trained over raw instances located in the hyper sphere 
            local_surogate, used_features = ape_tabular.lime_explainer.explain_instance_training_dataset(closest_counterfactual,
                                                                        ape_tabular.black_box_predict_proba, 
                                                                        num_features=nb_features_employed,#len(features_employed_in_rule),
                                                                        instances_in_sphere=instances_in_sphere,
                                                                        ape=ape_tabular,
                                                                        user_simulated=True)
            print("local surrogate explanation", local_surogate.as_list())
            """features_linear_employed = []
            for feature_linear_employed in local_surogate.as_list():
                features_linear_employed.append(feature_linear_employed[0])
            #print("features linear employed", features_linear_employed) => To print the features importance generated by Local Surrogate
            # Transform the explanation generated by Local Surrogate to know what are the features employed by LS
            rules, training_instances_pandas_frame, features_employed_by_extended_local_surrogate = ape_tabular.generate_rule_and_data_for_anchors(features_linear_employed, 
                                                                                                        ape_tabular.target_class, ape_tabular.train_data, 
                                                                                                        simulated_user_experiment=True)

            features_employed_by_extended_local_surrogate = list(set(features_employed_by_extended_local_surrogate))
            """
            features_employed_by_extended_local_surrogate = used_features
            return features_employed_by_extended_local_surrogate
        #else:
            # In case of multimodal data APE choose an anchor explanation 
            #return False
    # Always compute an anchor explanation to compare with APE and local Surrogate
    anchor_exp = ape_tabular.anchor_explainer.explain_instance(instance, ape_tabular.black_box_predict, threshold=ape_tabular.threshold_precision, 
                                delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, stop_on_first=False,
                                desired_label=None, beam_size=4)
    print("rule by anchor", anchor_exp.names())
    #print("rule by anchor", anchor_exp.names()) => To print the rules returned by Anchors
    # Transform the explanation generated by Anchors to know what are the features employed by Anchors
    rules, training_instances_pandas_frame, features_employed_in_rule = ape_tabular.generate_rule_and_data_for_anchors(anchor_exp.names(), 
                                                                                            ape_tabular.target_class, ape_tabular.train_data, 
                                                                                            simulated_user_experiment=True)
    features_employed_in_rule = list(set(features_employed_in_rule))
    if only_anchors:
        return features_employed_in_rule
    #print("rules in anchor", rules)
    if not ape_tabular.multimodal_results:
        # In case of unimodal data we compute a local surrogate explanation trained over raw instances located in the hyper sphere 
        local_surogate = ape_tabular.lime_explainer.explain_instance_training_dataset(closest_counterfactual,
                                                                    ape_tabular.black_box_predict_proba, 
                                                                    num_features=nb_features_employed,#len(features_employed_in_rule),
                                                                    instances_in_sphere=instances_in_sphere,
                                                                    ape=ape_tabular)
        #print("local surrogate explanation", local_surogate.as_list())
        features_linear_employed = []
        for feature_linear_employed in local_surogate.as_list():
            features_linear_employed.append(feature_linear_employed[0])
        #print("features linear employed", features_linear_employed) => To print the features importance generated by Local Surrogate
        # Transform the explanation generated by Local Surrogate to know what are the features employed by LS
        rules, training_instances_pandas_frame, features_employed_by_extended_local_surrogate = ape_tabular.generate_rule_and_data_for_anchors(features_linear_employed, 
                                                                                                    ape_tabular.target_class, ape_tabular.train_data, 
                                                                                                    simulated_user_experiment=True)

        features_employed_by_extended_local_surrogate = list(set(features_employed_by_extended_local_surrogate))
        features_employed_by_ape = features_employed_by_extended_local_surrogate
    else:
        # In case of multimodal data APE choose an anchor explanation 
        features_employed_by_ape = features_employed_in_rule
    
    # Generate a classic local Surrogate explanation model to compare with APE and Anchors
    local_surrogate = ape_tabular.lime_explainer.explain_instance(closest_counterfactual,
                                                                ape_tabular.black_box_predict_proba, 
                                                                num_features=nb_features_employed)#len(features_employed_in_rule))
    features_linear_employed = []
    #print("classic local surogate", local_surrogate.as_list())
    for feature_linear_employed in local_surrogate.as_list():
        features_linear_employed.append(feature_linear_employed[0])
    rules, training_instances_pandas_frame, features_employed_in_linear = ape_tabular.generate_rule_and_data_for_anchors(features_linear_employed, 
                                                                                                    ape_tabular.target_class, ape_tabular.train_data, 
                                                                                                    simulated_user_experiment=True)
    features_employed_in_linear = list(set(features_employed_in_linear))
    try:
        if features_employed_by_ape.sort() != features_employed_in_linear.sort():
            print("There is a difference between the features chosen by classic local Surrogate and the one chosen by APE")
            print("features employed by classic LS", features_employed_in_linear)
            print("features employed by APE", features_employed_by_ape)
            print("radiues", growing_sphere.radius)
            
        if features_employed_by_extended_local_surrogate.sort() != features_employed_in_linear.sort():
            print("There is a difference between the features chosen by classic local Surrogate and the one chosen by the Local Surrogate from APE")
            print("features employed by classic LS", features_employed_in_linear)
            print("features employed by extended Local Surrogate", features_employed_by_extended_local_surrogate)
    except:
        pass
    return features_employed_in_linear, features_employed_by_ape, features_employed_in_rule

def simulate_user_experiments_lime_ls(ape_tabular, instance, nb_features_employed, farthest_distance, closest_counterfactual, growing_sphere,
                              position_instances_in_sphere, nb_training_instance_in_sphere):
    """
    Function that generate a local surrogate explanation and a Lime explanation and 
    returns the features that are used by these explanation models
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: target instance to explain
          nb_features_employed: Number of features employed by the linear explanation model
          farthest_distance: Distance from the target instance to explain and the farthest instance from the training data
          closest_counterfactual: Closest instance from the closest counterfactual from the target instance (on the limit of the hyper field 
                                  from the same class as the target instance)
          growing_sphere: growing sphere object used by APE to explain the target instance
          position_instances_in_sphere: Index of the training data located in the hyper field
          nb_training_instance_in_sphere: Number of training data located in the hyper field
    Return: List of features employed by local surrogate and Lime in case of unimodal distribution
    """
    target_class = ape_tabular.black_box_predict(instance.reshape(1, -1))[0]
    ape_tabular.target_class = target_class
    
    """ Generates or store instances in the area of the hypersphere and their correspoinding labels """
    min_instance_per_class = ape_tabular.nb_min_instance_per_class_in_sphere
    position_instances_in_sphere, nb_training_instance_in_sphere = ape_tabular.instances_from_dataset_inside_sphere(closest_counterfactual, 
                                                            growing_sphere.radius, ape_tabular.train_data)

    instances_in_sphere, labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = ape_tabular.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                            closest_counterfactual, ape_tabular.train_data,
                                                                                                            farthest_distance, min_instance_per_class, 
                                                                                                            position_instances_in_sphere, 
                                                                                                            nb_training_instance_in_sphere, libfolding=True)
    
    if not ape_tabular.multimodal_results:
        # In case of unimodal distribution we compute a lime and a local surrogate models explanatios
        lime_exp = ape_tabular.lime_explainer.explain_instance(instance, ape_tabular.black_box_predict_proba, num_features=nb_features_employed)
        #print("Lime explanation", lime_exp.as_list())
        features_linear_employed = []
        for feature_linear_employed in lime_exp.as_list():
            features_linear_employed.append(feature_linear_employed[0])
        #print("features linear employed", features_linear_employed)
        rules, training_instances_pandas_frame, features_employed_in_linear = ape_tabular.generate_rule_and_data_for_anchors(features_linear_employed, 
                                                                                                    target_class, ape_tabular.train_data, 
                                                                                                    simulated_user_experiment=True)
        
        
        # Trained a local surrogate explanation model
        local_surrogate_exp = ape_tabular.lime_explainer.explain_instance_training_dataset(closest_counterfactual, 
                                                                            ape_tabular.black_box_predict_proba, 
                                                                            num_features=nb_features_employed, 
                                                                            instances_in_sphere=instances_in_sphere,
                                                                            ape=ape_tabular)
        features_local_surrogate_employed = []
        for feature_local_surrogate_employed in local_surrogate_exp.as_list():
            features_local_surrogate_employed.append(feature_local_surrogate_employed[0])
        rules, training_instances_pandas_frame, features_employed_in_local_surrogate = ape_tabular.generate_rule_and_data_for_anchors(features_local_surrogate_employed, 
                                                                                                    target_class, ape_tabular.train_data, 
                                                                                                    simulated_user_experiment=True)
        """
        counter_factual_class = ape_tabular.black_box_predict(closest_counterfactual.reshape(1,-1))[0]
        print("la classe du contre factuel le plus proche : ", counter_factual_class)
        print('Lime explanation for %s' % ape_tabular.class_names[target_class])
        print('\n'.join(map(str, lime_exp.as_list())))
        print('Local Surrogate explanation for %s' % ape_tabular.class_names[counter_factual_class])
        print('\n'.join(map(str, local_surrogate_exp.as_list())))
        """
        features_employed_in_linear.sort()
        features_employed_in_local_surrogate.sort()
        features_employed_in_linear = list(set(features_employed_in_linear)) 
        features_employed_in_local_surrogate = list(set(features_employed_in_local_surrogate))
        return features_employed_in_linear, features_employed_in_local_surrogate
    else:
        print("multimodal data")
        return [], []

def ape_illustrative_results(ape_tabular, instance, counterfactual_in_sphere):
    """
    Function that print the explanation of ape depending on the distribution of counterfactual instances located in the hyper field
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: Target instance to explain
          counterfactual_in_sphere: List of counterfactual instances located in the hyper field
    Return:
    """
    multimodal = ape_tabular.multimodal_results
    if multimodal:
        # In case of multimodal distribution we generate an anchor explanation with multiple clusters centers as counterfactual explanations
        anchor_exp = ape_tabular.anchor_explainer.explain_instance(instance, ape_tabular.black_box_predict, threshold=ape_tabular.threshold_precision, 
                                        delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, stop_on_first=False,
                                        desired_label=None, beam_size=4)
        # Generate rules and data frame for applying anchors 
        print("rule by anchor", anchor_exp.names())
        # Kelbow method to detect the best number of clusters
        visualizer = KElbowVisualizer(KMeans(), k=(1,8))
        x_elbow = np.array(counterfactual_in_sphere)
        visualizer.fit(x_elbow)
        n_clusters = visualizer.elbow_value_
        if n_clusters is not None:
            print("n CLUSTERS ", n_clusters) # Corresponding to the number of counterfactual instances generated by APE
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(counterfactual_in_sphere)
            clusters_centers = kmeans.cluster_centers_
            print("Mean center of clusters from KMEANS ", clusters_centers)
    else:
        # In case of unimodal distribution we generate a linear explanation model based on local surrogate trained over the training dataset along with
        # the closest counterfactual corresponding to the center of the explantion model
        local_surrogate_exp = ape_tabular.lime_explainer.explain_instance_training_dataset(ape_tabular.closest_counterfactual, 
                                                                    ape_tabular.black_box_predict_proba, num_features=4)
        counter_factual_class = ape_tabular.black_box_predict(ape_tabular.closest_counterfactual.reshape(1,-1))[0]
        print('Local Surrogate explanation for %s' % ape_tabular.class_names[counter_factual_class])
        print('\n'.join(map(str, local_surrogate_exp.as_list())))
        print("Closest counterfactual:", ape_tabular.closest_counterfactual)
    return 0, 0, 0

def modify_dataset(dataset, nb_feature_to_set_0, randomly=False):
    """
    Function that modify the dataset by turning values of data point for a given number of features to 0
    Args: dataset: The dataset that we want to modify
          nb_feature_to_set_0: Number of features for which we want to turn the values of dataset to 0
          randomly: parameter used to replace values in the dataset either by 0 (set to False) or random values (set to True)
    Return: A modified dataset 
    """
    # Create a list of values corresponding to the features that will be replaced to 0
    feature_modified = random.sample(range(0, len(dataset[0])), nb_feature_to_set_0)
    feature_kept = set(range(len(dataset[0]))).difference(feature_modified)
    dataset_to_return = dataset.copy()
    
    if randomly:
        # Used in case of replacing values with random values instead of 0
        for feature_modify in feature_modified:
            random_value = np.random.uniform(int(min(dataset[:,feature_modify])), int(max(dataset[:,feature_modify])), len(dataset)).tolist()
            dataset_to_return[:,feature_modify] = random_value 
    else:
        # Modify the dataset to replace values of the computed features by 0
        dataset_to_return[:,feature_modified] = 0
    print("feature kept", feature_kept)
    return dataset_to_return, feature_kept

def decision_tree_function(clf, instance):
    """
    Args: clf: Trained decision tree model
          instance: Target instance to explain
    Return: the set of features employed by the decision tree model
    """
    feature = clf.tree_.feature
    node_indicator = clf.decision_path(instance)
    leaf_id = clf.apply(instance)

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    
    #print('Rules used to predict sample {id}:\n'.format(id=sample_id))
    feature_employed = []
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue
        feature_employed.append(feature[node_id])
    return set(feature_employed)
