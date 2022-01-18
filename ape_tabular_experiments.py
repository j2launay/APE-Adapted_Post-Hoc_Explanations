import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from growingspheres import counterfactuals as cf
import random
from sklearn.metrics import accuracy_score
from growingspheres.utils.gs_utils import distances


def model_stability_index(ape_explainer, instance, growing_method, opponent_class, n_instance_per_layer, first_radius, 
                        dicrease_radius, farthest_distance):
    """
    Compute the msi values (model stability index) to discover whether APE is returning the same type of explanation multiple times 
    for a same target instance
    Args: instance: Target instance on which we want to use APE several times
            growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
            opponent_class: Class of the closest counterfactual
            n_instance_per_layer: Hyperparameter of the number of instances we want to generate for growing_method
            first_radius: Hyperparameter to initalize the radius of the growing_method
            dicrease_radius: Hyperparameter to indicate the speed to dicrease the radius of the growing_method
            farthest_distance: the distance between the target instance and its farthest instance from the training dataset
    Return: The type of explanation that APE indicates as the more suitable
    """
    # Find the closest counterfactual instance close to the target instance
    growing_fields = cf.CounterfactualExplanation(instance, ape_explainer.black_box_predict, method=growing_method, 
                    target_class=opponent_class, continuous_features=ape_explainer.continuous_features, 
                    categorical_features=ape_explainer.categorical_features, 
                    categorical_values=ape_explainer.categorical_values, max_features=ape_explainer.max_features,
                    min_features=ape_explainer.min_features)
    growing_fields.fit(n_in_layer=n_instance_per_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=True, 
                verbose=ape_explainer.verbose, feature_variance=ape_explainer.feature_variance, 
                farthest_distance_training_dataset=farthest_distance, 
                probability_categorical_feature=ape_explainer.probability_categorical_feature, 
                min_counterfactual_in_field=ape_explainer.nb_min_instance_per_class_in_field)
    closest_counterfactual = growing_fields.enemy
    # Compute the farthest distance between the counterfactual and real instances from the training dataset
    farthest_distance_cf = 0
    for training_instance in ape_explainer.train_data:
        # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
        # but it adds a sparsity distance computation (i.e: number of same values) 
        farthest_distance_cf_now = distances(ape_explainer.closest_counterfactual, training_instance, ape_explainer)
        if farthest_distance_cf_now > farthest_distance_cf:
            farthest_distance_cf = farthest_distance_cf_now

    # Generate or store instances in the area of the hyperfield and their corresponding labels
    min_instance_per_class = ape_explainer.nb_min_instance_per_class_in_field
    position_instances_in_field, nb_training_instance_in_field = ape_explainer.instances_from_dataset_inside_field(closest_counterfactual, 
                                                                                                            growing_fields.radius, ape_explainer.train_data)

    instances_in_field, _, _, instances_in_field_libfolding = ape_explainer.generate_instances_inside_field(growing_fields.radius, 
                                                                                                            closest_counterfactual, ape_explainer.train_data, 
                                                                                                            farthest_distance_cf, min_instance_per_class, 
                                                                                                            position_instances_in_field, 
                                                                                                            nb_training_instance_in_field, 
                                                                                                            libfolding=True,
                                                                                                            growing_method=growing_method)
    # Compute the libfolding test to verify wheter instances in the area of the hyper field is multimodal or unimodal
    unimodal_test = ape_explainer.check_test_unimodal_data(instances_in_field, instances_in_field_libfolding)
    while not unimodal_test:
        # While the libfolding test is not able to declare that data are multimodal or unimodal we extend the number of instances that are generated
        min_instance_per_class *= 1.5
        instances_in_field, _, _, instances_in_field_libfolding = ape_explainer.generate_instances_inside_field(growing_fields.radius, 
                                                                                                closest_counterfactual, ape_explainer.train_data, farthest_distance_cf, 
                                                                                                min_instance_per_class, position_instances_in_field, 
                                                                                                nb_training_instance_in_field, libfolding=True,
                                                                                                            growing_method=growing_method)
        
        unimodal_test = ape_explainer.check_test_unimodal_data(instances_in_field, instances_in_field_libfolding)
    return ape_explainer.multimodal_results

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
        local_surrogate_exp = ape.linear_explainer.explain_instance_training_dataset(closest_counterfactual, 
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
    """ Computes the labels for instances from the dataset to compute accuracy for explanation method """
    labels_instance_test_data = ape.black_box_predict(ape.test_data)
    nb_instance_test_data_label_as_target = sum(x == ape.target_class for x in labels_instance_test_data)
    
    
    anchor_accuracy, anchor_coverage, anchor_f2, anchor_explanation = ape.compute_anchor_accuracy_coverage(instance, 
                                    labels_instance_test_data, len(test_instances_in_sphere), 
                                    farthest_distance, growing_method=growing_method)
    ls_accuracy, ls_coverage, ls_f2, ls_explanation, ape.extended_radius = ape.compute_lime_extending_accuracy_coverage(training_instances_in_sphere, 
                                    instance, train_labels_in_sphere, growing_sphere, nb_features_employed,
                                    farthest_distance, growing_method)
    if ape.multimodal_results:
        # In case of multimodal data, we generate a rule based explanation and compute accuracy and coverage of this explanation model
        ape_accuracy, ape_coverage, ape_f2, ape_explanation = anchor_accuracy, anchor_coverage, anchor_f2, anchor_explanation

    else:
        # In case of unimodal data, we generate linear explanation and compute accuracy and coverage of this explanation model
        ape_accuracy, ape_coverage, ape_f2, ape_explanation = ls_accuracy, ls_coverage, ls_f2, ls_explanation
    return [ls_accuracy, anchor_accuracy, ape_accuracy], [ls_coverage, anchor_coverage, ape_coverage], [ls_f2, anchor_f2, ape_f2], \
                ape.multimodal_results, ape.extended_radius, ape.separability_index, ape.pvalue 

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
            local_surogate, used_features = ape_tabular.linear_explainer.explain_instance_training_dataset(closest_counterfactual,
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
        local_surogate = ape_tabular.linear_explainer.explain_instance_training_dataset(closest_counterfactual,
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
    local_surrogate = ape_tabular.linear_explainer.explain_instance(closest_counterfactual,
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
        lime_exp = ape_tabular.linear_explainer.explain_instance(instance, ape_tabular.black_box_predict_proba, num_features=nb_features_employed)
        #print("Lime explanation", lime_exp.as_list())
        features_linear_employed = []
        for feature_linear_employed in lime_exp.as_list():
            features_linear_employed.append(feature_linear_employed[0])
        #print("features linear employed", features_linear_employed)
        rules, training_instances_pandas_frame, features_employed_in_linear = ape_tabular.generate_rule_and_data_for_anchors(features_linear_employed, 
                                                                                                    target_class, ape_tabular.train_data, 
                                                                                                    simulated_user_experiment=True)
        
        
        # Trained a local surrogate explanation model
        local_surrogate_exp = ape_tabular.linear_explainer.explain_instance_training_dataset(closest_counterfactual, 
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
