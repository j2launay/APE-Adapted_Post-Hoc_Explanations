import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from growingspheres import counterfactuals as cf
from sklearn.neighbors import NearestNeighbors
import random
from sklearn.metrics import accuracy_score
from growingspheres.utils.gs_utils import generate_inside_ball, generate_inside_field, \
    generate_categoric_inside_ball, distances
#from anchors.limes.utils_stability import compare_confints

def k_closest_experiments(ape_explainer, k_closest, metrics):
    """
    Compute the average distance to the k closest counterfactual from the real dataset to the artificial counterfactual generated by the growing method
    Args: ape_explainer
          k_closest: Number of k closest points to compute the distance to the closest counterfactual
    Return: Average distance to the k closest counterfactual
            Mean of the distance to the whole counterfactual instances from the training set?
    """
    index_train_data_counterfactual_class = np.where([x != ape_explainer.target_class for x in ape_explainer.black_box_predict(ape_explainer.train_data)])
    index_train_data_target_class = list(set(range(0, ape_explainer.train_data.shape[0])) - set(index_train_data_counterfactual_class[0]))

    train_data_counterfactual_class, train_data_target_class = ape_explainer.train_data[index_train_data_counterfactual_class], ape_explainer.train_data[index_train_data_target_class]
    avg_dists = []
    
    print("max mahalanobis", ape_explainer.max_mahalanobis)
    set_set_true_instances = [ape_explainer.test_data, train_data_counterfactual_class, train_data_target_class]
    for set_true_instances in set_set_true_instances:
        if len(set_true_instances) == 0:
            avg_dists.append(None)
        else:
            try:
                temp_distance = []
                for cluster_center in ape_explainer.clusters_centers:
                    temp_distance.append(distances(cluster_center, set_true_instances, ape=ape_explainer, metrics='mahalanobis', dataset=set_true_instances))
                avg_dists.append(abs(np.mean(temp_distance)))
            except Exception as inst:
                avg_dists.append(abs(distances(ape_explainer.closest_counterfactual, set_true_instances, ape=ape_explainer, metrics='mahalanobis', dataset=set_true_instances)))
    print("mahalanobis done", avg_dists)
    
    metrics = ['w_manhattan', 'w_euclidian']
    for metric in metrics:
        farthest_distance_cf = 0
        for training_instance in ape_explainer.train_data:
            farthest_distance_cf_now = distances(ape_explainer.closest_counterfactual, training_instance, ape_explainer, metrics=metric)
            if farthest_distance_cf_now > farthest_distance_cf:
                farthest_distance_cf = farthest_distance_cf_now
        ape_explainer.farthest_distance = farthest_distance_cf
        
        neigh = NearestNeighbors(n_neighbors=k_closest+1, algorithm='ball_tree', metric=distances, metric_params={"ape": ape_explainer, "metrics": metric})
        neigh.fit(train_data_counterfactual_class[:1000], ape_explainer.black_box_predict(train_data_counterfactual_class[:1000]))
        try:
            dists, neighs = neigh.kneighbors(ape_explainer.clusters_centers, k_closest+1)
        except AttributeError:
            dists, neighs = neigh.kneighbors(ape_explainer.closest_counterfactual.reshape(1, -1), k_closest+1)
        closest_ennemis = train_data_counterfactual_class[neighs[0]]
        dists = []
        for ennemis in closest_ennemis:
            dists.append(distances(ape_explainer.closest_counterfactual, ennemis, ape_explainer, metrics=metric))
        #mean_dists, _ = neigh.kneighbors(train_data_counterfactual_class[:500], k_closest+1)
        avg_dists.append(np.mean(dists))
        print(metric, "done")
    #mean_avg_dists = np.mean(mean_dists[:, 1:], axis=1)
    
    return avg_dists, None#np.mean(mean_avg_dists)

def temp_model_stability_index(ape_explainer, instance, growing_method, opponent_class, n_instance_per_layer, first_radius, 
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

def model_stability_index(ape_explainer, instance, growing_method, opponent_class, n_instance_per_layer, first_radius, 
                        dicrease_radius, farthest_distance, nb_test=10):
    """
    Compute the msi (model stability index), csi and vsi values to discover whether APE is returning the same type of explanation multiple times 
    for a same target instance
    Args: ape_explainer: ape_tabular object that already computed closest counterfactual and sampled artificial instances 
            instance: Target instance on which we want to use APE several times
            growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
            opponent_class: Class of the closest counterfactual
            n_instance_per_layer: Hyperparameter of the number of instances we want to generate for growing_method
            first_radius: Hyperparameter to initalize the radius of the growing_method
            dicrease_radius: Hyperparameter to indicate the speed to dicrease the radius of the growing_method
            farthest_distance: the distance between the target instance and its farthest instance from the training dataset
            nb_test: The number of time we want to test if the explanation returns the same explanation
    Return: The type of explanation that APE indicates as the more suitable
    """
    print("compute model stability index...")
    model_stability = []
    initial_multimodal = ape_explainer.multimodal_results
    model_stability.append(initial_multimodal)
    for i in range(nb_test):
        model_stability.append(temp_model_stability_index(ape_explainer, instance, growing_method, 
                                opponent_class, n_instance_per_layer, first_radius, 
                                dicrease_radius, farthest_distance))
    model_stability_score =  model_stability.count(initial_multimodal) / (nb_test+1)

    csi_ls, vsi_ls = ape_explainer.linear_explainer.check_stability(ape_explainer.closest_counterfactual, \
        ape_explainer.black_box_predict_proba, n_calls=nb_test, index_verbose=False)

    confidence_intervals = []
    for i in range(nb_test):
        features_employed_in_rule = return_anchors_features(ape_explainer, instance)
        confidence_intervals.append(compute_vsi_anchors(ape_explainer, features_employed_in_rule))
    csi_anchors, vsi_anchors = compare_confints(confidence_intervals=confidence_intervals,
                            index_verbose=True, anchors=True)
    print("csi / vsi anchor", csi_anchors, vsi_anchors)

    if ape_explainer.multimodal_results:
        vsi = [vsi_ls, vsi_anchors, vsi_anchors]
        csi = [csi_ls, None, None]
    else:
        vsi = [vsi_ls, vsi_anchors, vsi_ls]
        csi = [csi_ls, None, None]
    return model_stability_score, csi, vsi


def compute_vsi_anchors(ape_explainer, used_features):
    feature_ids = used_features
    used_features = [ape_explainer.feature_names[i] for i in feature_ids]
    conf_int = {}
    for test in used_features:
        conf_int[test] = 1
    return conf_int


def return_anchors_features(ape_tabular, instance):
    """
    Function that generate an anchor explanation and returns the features that are used by this anchor
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: target instance to explain
    Return: List of features employed by Anchors
    """
    target_class = ape_tabular.black_box_predict(instance.reshape(1, -1))[0]
    
    anchor_exp = ape_tabular.anchor_explainer.explain_instance(instance, ape_tabular.black_box_predict, threshold=ape_tabular.threshold_precision, 
                                delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, stop_on_first=False,
                                desired_label=None, beam_size=4)

    #print("rule by anchor", anchor_exp.names()) => To print the rules returned by Anchors
    # Transform the explanation generated by Anchors to know what are the features employed by Anchors
    _, _, features_employed_in_rule = ape_tabular.generate_rule_and_data_for_anchors(anchor_exp.names(), 
                                                                                            target_class, ape_tabular.train_data, 
                                                                                            simulated_user_experiment=True)
    return list(set(features_employed_in_rule))

def compute_k_radius_local_surrogate_extending(ape_tabular, instance, growing_field, nb_instance_test_data_label_as_target,
                                                instances_in_field, labels_in_field, farthest_distance_cf, nb_features_employed,
                                                growing_method, linear_model, range_radius):
    accuracys_local_surrogate = []
    def generate_inside_field_radius(dataset, radius):
            position_training_instances_in_field, nb_training_instance_in_field = ape_tabular.instances_from_dataset_inside_field(instance, 
                                                                                                                radius, dataset)
            try:
                if growing_method == "GS":
                    generated_instances_inside_field = generate_inside_ball(instance, (0, radius), 
                                        max(1, int (ape_tabular.nb_min_instance_in_field - nb_training_instance_in_field)))
                elif len(ape_tabular.categorical_features) > 1:
                    generated_instances_inside_field = generate_categoric_inside_ball(instance, (0, radius), radius,
                                                            max(1, int (ape_tabular.nb_min_instance_in_field - nb_training_instance_in_field)), 
                                                            ape_tabular.continuous_features, ape_tabular.categorical_features, 
                                                            ape_tabular.categorical_values, feature_variance=ape_tabular.feature_variance,
                                                            probability_categorical_feature=ape_tabular.probability_categorical_feature,
                                                            min_features=ape_tabular.min_features, max_features=ape_tabular.max_features)
                else:
                    generated_instances_inside_field = generate_inside_field(instance, (0, radius), 
                                                        max(1, int (ape_tabular.nb_min_instance_in_field - nb_training_instance_in_field)), 
                                                        feature_variance=ape_tabular.feature_variance, 
                                                        min_features=ape_tabular.min_features, max_features=ape_tabular.max_features)
            except OverflowError:
                print("over flow error")

            instances_in_field = np.append(dataset[position_training_instances_in_field], generated_instances_inside_field, axis=0) if position_training_instances_in_field != [] else generated_instances_inside_field
            return instances_in_field, ape_tabular.black_box_predict(instances_in_field)

    for k in range_radius:
        print("computing for radius", k)
        train_instances_in_field, train_labels_in_field = generate_inside_field_radius(ape_tabular.train_data, k)
        test_instances_in_field, test_labels_in_field = generate_inside_field_radius(ape_tabular.test_data, k)
        
        # Train a Local Surrogate explanation model on a larger hyper field (with instances inside this hyper field)
        ls_raw_data = ape_tabular.linear_explainer.explain_instance_training_dataset(instance, ape_tabular.black_box_predict_proba, 
                                                                num_features=nb_features_employed,
                                                                model_regressor=linear_model,
                                                                instances_in_sphere=train_instances_in_field,
                                                                ape=ape_tabular)
        
        # Compute the accuracy of the new Local Surrogate and replace the accuracy score of the model if it is better than the old one
        prediction_inside_field = ape_tabular.modify_instance_for_linear_model(ls_raw_data, test_instances_in_field)
        accuracy_ls_raw_data = {}
        accuracy_ls_raw_data["all"], accuracy_ls_raw_data["target"], accuracy_ls_raw_data["counterfactual"], accuracy_ls_raw_data["auc"], \
            accuracy_ls_raw_data["auc_target"], accuracy_ls_raw_data["auc_counterfactual"]  = ape_tabular.compute_linear_regression_accuracy(prediction_inside_field, test_labels_in_field,\
                ls_raw_data.easy_model.intercept_)
        accuracys_local_surrogate.append(accuracy_ls_raw_data["all"])
        accuracys_local_surrogate.append(accuracy_ls_raw_data["counterfactual"])
        accuracys_local_surrogate.append(accuracy_ls_raw_data["auc"])
        accuracys_local_surrogate.append(accuracy_ls_raw_data["auc_counterfactual"])
    return accuracys_local_surrogate, growing_field.radius

def compute_all_explanation_method_accuracy(ape_tabular, instance, growing_field, nb_instance_test_data_label_as_target,
                                                position_instances_in_field, instances_in_field, labels_in_field,
                                                farthest_distance_cf, nb_features_employed,
                                                growing_method, linear_model):
    """
    Compute Precision, Coverage and F1 for APE, Anchors and the best Local Surrogate explanation models
    Args: ape_tabular: Ape tabular object used to explain the target instance
            instance: Target instance to explain
            growing_field: Growing Sphere object used by APE
            nb_instance_test_data_label_as_target: Number of instances from the testing data that are label as the target instance
            position_instances_in_field: Index of the instances from the training data located in the hyper field
            instances_in_field: Set of instances (artificial and training data) located in the hyper field
            labels_in_field: Classification of training instances in the hyper field by the black box model
            farthest_distance: Distance from the target instance to the farthest instance from the training data
            nb_features_employed: Number of features employed by the linear explanation model
            growing_method: The method used to sample artificial instances (between growing spheres and growing fields)
    Return: Arrays of accuracy, coverage and F1 of multiple explanation models
    """
    labels_instance_test_data = ape_tabular.black_box_predict(ape_tabular.test_data)
    ape_tabular.instance_to_explain = instance
    print("dans ape experiments functions")
    # Compute accuracy, coverage and F2 of local surrogate trained over raw data, with an extending field and a logistic regression model as explanation
    local_surrogate_extend_raw_accuracy, local_surrogate_initial_raw_accuracy, local_surrogate_extend_raw_coverage, f2_local_surrogate_extend_raw, _, \
                        ape_tabular.extended_radius = ape_tabular.compute_lime_extending_accuracy_coverage(instances_in_field, 
                                            ape_tabular.closest_counterfactual, labels_in_field, growing_field, nb_features_employed, 
                                            farthest_distance_cf, growing_method, position_instances_in_field, linear_model)
    # Compute accuracy, coverage and F2 for Anchors
    anchor_accuracy, anchor_coverage, f2_anchor, _ = ape_tabular.compute_anchor_accuracy_coverage(instance, 
                                    labels_instance_test_data, len(instances_in_field), 
                                    nb_instance_test_data_label_as_target,
                                    growing_method)
    # Compute accuracy, coverage and F2 for Decision Tree
    decision_tree_accuracy, decision_tree_coverage, f2_decision_tree, _, \
                decision_tree_radius = ape_tabular.compute_decision_tree_accuracy_coverage(instances_in_field, labels_in_field, 
                                        ape_tabular.closest_counterfactual, growing_field.radius, position_instances_in_field)
    # Compute accuracy for classic Local Surrogate
    local_surrogate = ape_tabular.linear_explainer.explain_instance(ape_tabular.closest_counterfactual,
                                                                ape_tabular.black_box_predict_proba,
                                                                model_regressor=linear_model,
                                                                num_features=nb_features_employed)
    local_surrogate_instances, _ = ape_tabular.linear_explainer.data_inverse(ape_tabular.closest_counterfactual,len(instances_in_field))
    local_surrogate_prediction = ape_tabular.modify_instance_for_linear_model(local_surrogate, local_surrogate_instances)
    local_surrogate_labels = ape_tabular.black_box_predict(local_surrogate_instances)
    accuracy_local_surrogate = {'real':None}
    accuracy_local_surrogate['all'], accuracy_local_surrogate["target"], accuracy_local_surrogate["counterfactual"], accuracy_local_surrogate["auc"], \
                accuracy_local_surrogate["auc_target"], accuracy_local_surrogate["auc_counterfactual"]  = \
                    ape_tabular.compute_linear_regression_accuracy(local_surrogate_prediction, local_surrogate_labels, local_surrogate.easy_model.intercept_)

    # Compute coverage and F2 for classic Local Surrogate
    position_testing_instances_in_field, nb_testing_instance_in_field = ape_tabular.instances_from_dataset_inside_field(ape_tabular.closest_counterfactual, 
                                                                                                                growing_field.radius, ape_tabular.test_data)
    nb_testing_instance_in_field_label_as_target, _ = ape_tabular.compute_labels_inside_field(nb_testing_instance_in_field, 
                                                                                                                position_testing_instances_in_field,
                                                                                                                ape_tabular.test_data)
    linear_coverage = nb_testing_instance_in_field_label_as_target/nb_instance_test_data_label_as_target
    f2_linear_surrogate = (2*accuracy_local_surrogate['all'] + linear_coverage)/3
    
    # Compute accuracy for LIME
    lime = ape_tabular.linear_explainer.explain_instance(instance, ape_tabular.black_box_predict_proba,
                                                                model_regressor=linear_model,
                                                                num_features=nb_features_employed)
    lime_instances, _ = ape_tabular.linear_explainer.data_inverse(instance,len(instances_in_field))
    lime_prediction = ape_tabular.modify_instance_for_linear_model(lime, lime_instances)
    lime_labels = ape_tabular.black_box_predict(lime_instances)
    accuracy_lime = {'real':None}
    accuracy_lime['all'], accuracy_lime["target"], accuracy_lime["counterfactual"], accuracy_lime["auc"], \
                accuracy_lime["auc_target"], accuracy_lime["auc_counterfactual"]  = \
                    ape_tabular.compute_linear_regression_accuracy(lime_prediction, lime_labels, lime.easy_model.intercept_)

    # Compute coverage and F2 for LIME
    position_testing_instances_in_field, nb_testing_instance_in_field = ape_tabular.instances_from_dataset_inside_field(instance, 
                                                                                                                growing_field.radius, ape_tabular.test_data)
    nb_testing_instance_in_field_label_as_target, _ = ape_tabular.compute_labels_inside_field(nb_testing_instance_in_field, 
                                                                                                                position_testing_instances_in_field,
                                                                                                                ape_tabular.test_data)
    lime_coverage = nb_testing_instance_in_field_label_as_target/nb_instance_test_data_label_as_target
    f2_lime = (2*accuracy_lime['all'] + linear_coverage)/3

    # Select values for APE depending on the unimodality test
    apea_accuracy = anchor_accuracy if ape_tabular.multimodal_results else local_surrogate_extend_raw_accuracy
    apea_coverage = anchor_coverage if ape_tabular.multimodal_results else local_surrogate_extend_raw_coverage
    f2_apea = f2_anchor if ape_tabular.multimodal_results else f2_local_surrogate_extend_raw
    apet_accuracy = decision_tree_accuracy if ape_tabular.multimodal_results else local_surrogate_extend_raw_accuracy
    apet_coverage = decision_tree_coverage if ape_tabular.multimodal_results else local_surrogate_extend_raw_coverage
    f2_apet = f2_decision_tree if ape_tabular.multimodal_results else f2_local_surrogate_extend_raw

    accuracys = [accuracy_local_surrogate['all'], accuracy_local_surrogate["target"], accuracy_local_surrogate["counterfactual"], accuracy_local_surrogate["auc"], \
                accuracy_local_surrogate["auc_target"], accuracy_local_surrogate["auc_counterfactual"], local_surrogate_initial_raw_accuracy['all'],\
                local_surrogate_initial_raw_accuracy["target"], local_surrogate_initial_raw_accuracy["counterfactual"], local_surrogate_initial_raw_accuracy["auc"], \
                local_surrogate_initial_raw_accuracy["auc_target"], local_surrogate_initial_raw_accuracy["auc_counterfactual"], \
                local_surrogate_extend_raw_accuracy['all'], local_surrogate_extend_raw_accuracy["target"], local_surrogate_extend_raw_accuracy["counterfactual"], \
                local_surrogate_extend_raw_accuracy["auc"], local_surrogate_extend_raw_accuracy["auc_target"], \
                local_surrogate_extend_raw_accuracy["auc_counterfactual"], accuracy_lime['all'], accuracy_lime["target"], accuracy_lime["counterfactual"], \
                accuracy_lime["auc"], accuracy_lime["auc_target"], accuracy_lime["auc_counterfactual"], anchor_accuracy['all'], decision_tree_accuracy['all'], \
                apea_accuracy['all'], apet_accuracy['all']]
    
    coverages = [linear_coverage, linear_coverage, linear_coverage, linear_coverage, linear_coverage, linear_coverage, linear_coverage, \
        linear_coverage, linear_coverage, linear_coverage, linear_coverage, linear_coverage, \
        linear_coverage, linear_coverage, linear_coverage, linear_coverage, linear_coverage, \
        local_surrogate_extend_raw_coverage, lime_coverage, lime_coverage, lime_coverage, lime_coverage, lime_coverage, lime_coverage, \
        anchor_coverage, decision_tree_coverage, apea_coverage, apet_coverage]
    
    f2s = [f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, \
        f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, \
        f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, f2_linear_surrogate, \
         f2_local_surrogate_extend_raw, f2_lime, f2_lime, f2_lime, f2_lime, f2_lime, f2_lime, f2_anchor, f2_decision_tree, f2_apea, f2_apet]
    
    accuracys_real = [accuracy_local_surrogate['real'], accuracy_local_surrogate['real'], accuracy_local_surrogate['real'], \
            accuracy_local_surrogate['real'], accuracy_local_surrogate['real'], accuracy_local_surrogate['real'], \
            local_surrogate_initial_raw_accuracy['real'], local_surrogate_initial_raw_accuracy['real target'], \
            local_surrogate_initial_raw_accuracy['real counterfactual'], local_surrogate_initial_raw_accuracy['real auc'], \
            local_surrogate_initial_raw_accuracy['real auc_target'], local_surrogate_initial_raw_accuracy['real auc_counterfactual'], \
            local_surrogate_extend_raw_accuracy['real'], local_surrogate_extend_raw_accuracy['real target'], local_surrogate_extend_raw_accuracy['real counterfactual'], \
            local_surrogate_extend_raw_accuracy['real auc'], local_surrogate_extend_raw_accuracy['real auc_target'], \
            local_surrogate_extend_raw_accuracy['real auc_counterfactual'], None, None, None, None, None, None, \
            anchor_accuracy['real'], decision_tree_accuracy['real'], apea_accuracy['real'], apet_accuracy['real']]
    multimodal = 1 if ape_tabular.multimodal_results else 0
    return accuracys, coverages, f2s, multimodal, ape_tabular.extended_radius, accuracys_real

def compute_local_surrogate_accuracy_coverage(ape_tabular, instance, growing_field,
                                                test_instances_in_field, test_labels_in_field,
                                                position_testing_instances_in_field, nb_testing_instance_in_field, 
                                                nb_features_employed, growing_method):
    """
    Compute accuracy, coverage and f2 for multiple linear explanation models
    Args: ape_tabular: ape tabular object used to explain the target instance
          instance: target instance to explain
          growing field: growing field object used by APE
          instances_in_field: (artificial and trained) data located in the hyper field
          labels_in_field: Labels by the black box model of the instances located in the hyper field
          position_instances_in_field: Index of the instaces from the training data located in the field
          nb_training_instance_in_field: Number of training data located in the hyper field
          nb_features_employed: Number of features employed by the linear explanations models
    Return: accuracy, coverage and f2 of classic local surrogate, a local surrogate train over training data with a linear regression 
            and a local surrogate using a logistic regression model as explanation
    """
    labels_instance_test_data = ape_tabular.black_box_predict(ape_tabular.test_data)
    nb_instance_test_data_label_as_target = sum(x == ape_tabular.target_class for x in labels_instance_test_data)
    
    nb_testing_instance_in_field_label_as_target, labels_testing_instance_in_field = ape_tabular.compute_labels_inside_field(nb_testing_instance_in_field, 
                                                                                                                                position_testing_instances_in_field,
                                                                                                                                ape_tabular.test_data)    
    
    # Trained a local surrogate model over binarize data using a linear regression (default Local Surrogate)
    local_surrogate = ape_tabular.linear_explainer.explain_instance(ape_tabular.closest_counterfactual, 
                                                                ape_tabular.black_box_predict_proba, 
                                                                num_features=nb_features_employed)
    prediction_inside_field = ape_tabular.modify_instance_for_linear_model(local_surrogate, test_instances_in_field)
    accuracy_local_surrogate, accuracy_local_surrogate_target, accuracy_local_surrogate_counterfactual, accuracy_local_surrogate_auc, \
                accuracy_local_surrogate_auc_target, accuracy_local_surrogate_auc_counterfactual = \
                    ape_tabular.compute_linear_regression_accuracy(prediction_inside_field, labels_testing_instance_in_field, local_surrogate.easy_model.intercept_)

    # Trained a local surrogate model over training data using a linear regression (by default on Lime)
    local_surogate_raw_data = ape_tabular.linear_explainer.explain_instance_training_dataset(ape_tabular.closest_counterfactual, 
                                                                ape_tabular.black_box_predict_proba, num_features=nb_features_employed,
                                                                instances_in_sphere=test_instances_in_field,
                                                                ape=ape_tabular)
    prediction_inside_field = ape_tabular.modify_instance_for_linear_model(local_surogate_raw_data, test_instances_in_field)
    accuracy_local_surogate_raw_data, accuracy_local_surrogate_target_raw_data, accuracy_local_surrogate_counterfactual_raw_data, \
        accuracy_local_surrogate_auc_raw_data, accuracy_local_surrogate_auc_target_raw_data, \
            accuracy_local_surrogate_auc_counterfactual_raw_data = ape_tabular.compute_linear_regression_accuracy(prediction_inside_field, \
                labels_testing_instance_in_field, local_surogate_raw_data.easy_model.intercept_)
    
    # Compute accuracy for a local Surrogate model with a Logistic Regression model as explanation model over binary data
    local_surrogate_exp_regression = ape_tabular.linear_explainer.explain_instance(ape_tabular.closest_counterfactual,  
                                                            ape_tabular.black_box_predict, num_features=nb_features_employed,
                                                            model_regressor = LogisticRegression())
    prediction_inside_field = ape_tabular.modify_instance_for_linear_model(local_surrogate_exp_regression, test_instances_in_field)
    accuracy_local_surrogate_logistic_regression = max(accuracy_score(labels_testing_instance_in_field, prediction_inside_field),
                                                        accuracy_score(labels_testing_instance_in_field, prediction_inside_field)) 

    # Compute accuracy, coverage and f2 for Extended Local Surrogate
    accuracy_ls_raw_data, accuracy_ls_raw_data_not_extending, lime_extending_coverage, f2_lime_extending, ls_explanation, radius = ape_tabular.compute_lime_extending_accuracy_coverage(
                                                test_instances_in_field,
                                                ape_tabular.closest_counterfactual, labels_testing_instance_in_field, growing_field, 
                                                nb_features_employed, 1, growing_method, position_testing_instances_in_field)

    if ape_tabular.verbose: print("Computing multiple linear explanation models accuracy and coverage.")
    linear_coverage = nb_testing_instance_in_field_label_as_target/nb_instance_test_data_label_as_target
    f2_lime_regression = (2*accuracy_local_surrogate_logistic_regression+linear_coverage)/3
    f2_not_bin_lime = (2*accuracy_local_surogate_raw_data+linear_coverage)/3
    f2_local_surrogate = (2*accuracy_local_surrogate+linear_coverage)/3
    
    return [accuracy_local_surrogate, accuracy_local_surrogate_logistic_regression, accuracy_local_surogate_raw_data, accuracy_ls_raw_data['all']], \
                [linear_coverage, linear_coverage, linear_coverage, lime_extending_coverage], \
                    [f2_local_surrogate, f2_lime_regression, f2_not_bin_lime, f2_lime_extending]
    