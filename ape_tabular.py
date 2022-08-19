import copy
import re
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer
from anchors import anchor_tabular, limes
from ape_experiments_functions import k_closest_experiments, compute_all_explanation_method_accuracy, \
        simulate_user_experiments
from growingspheres import counterfactuals as cf
from growingspheres.utils.gs_utils import generate_inside_ball, generate_inside_field, generate_categoric_inside_ball, \
        distances
import pyfolding as pf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import time
import random

class ApeTabularExplainer(object):
    """
    Args:
    """
    def __init__(self, train_data, class_names, black_box_predict, black_box_predict_proba=None,
                multiclass = False, continuous_features=None, categorical_features=None,
                categorical_values = None, feature_names=None, discretizer="quartile", 
                nb_min_instance_in_field=800, threshold_precision=0.95, 
                nb_min_instance_per_class_in_field=100, verbose=False, 
                categorical_names=None, linear_separability_index=0.99,
                transformations=None):
        self.feature_names = feature_names
        self.max_mahalanobis = None
        # We first split the training set into 60% train and 40% test in order to compute accuracy and coverage over test data
        self.train_data, self.test_data = train_test_split(train_data, test_size=0.4, random_state=42)
        test_mahanalobis = distances(self.train_data, self.train_data, self, metrics="mahalanobis")
        self.max_mahalanobis = max(test_mahanalobis['mahalanobis'])

        self.class_names = class_names
        self.black_box_predict = lambda x: black_box_predict(x)
        # black box predict proba is used for lime explanation with probabilistic function
        if black_box_predict_proba is not None:
            self.black_box_predict_proba = lambda x: black_box_predict_proba(x)
        self.categorical_names = categorical_names
        self.multiclass = multiclass
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.feature_names = feature_names
        self.discretizer = discretizer
        self.nb_min_instance_in_field = nb_min_instance_in_field
        self.threshold_precision = threshold_precision
        self.nb_min_instance_per_class_in_field = nb_min_instance_per_class_in_field
        self.verbose = verbose
        self.linear_separability_index = linear_separability_index
        self.black_box_labels = black_box_predict(self.train_data)
        if self.verbose: print("Setting interpretability methods")
        self.anchor_explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, self.train_data, 
                                                                    copy.copy(categorical_names), discretizer="MDLP", 
                                                                    black_box_labels=self.black_box_labels, ordinal_features=continuous_features)
        
        if categorical_features != []:
            # We need to fit a one hot encoder in order to generate linear explanation over one hot encoded data
            self.transformations = transformations
            self.enc = OneHotEncoder(handle_unknown='ignore')
            train_enc = self.train_data[:,categorical_features]
            self.enc.fit(train_enc)
            codes = self.enc.transform(train_enc).toarray()
            categorical_features_names = []
            for i in categorical_features:
                categorical_features_names.append(feature_names[i])
            oec_train_data = np.append(np.asarray(codes), self.train_data[:,continuous_features], axis=1)
            self.encoded_features_names = self.enc.get_feature_names(categorical_features_names)
            lime_features_names = []
            lime_categorical_features = []
            for i in range(len(oec_train_data[0])):
                if i < len(continuous_features):
                    lime_features_names.append(feature_names[continuous_features[i]])
                else:
                    lime_categorical_features.append(i)
            self.encoded_features_names = np.append(lime_features_names,[x for x in self.encoded_features_names]) .tolist()
            
        
        self.linear_explainer = limes.lime_tabular.LimeTabularExplainer(self.train_data, feature_names=feature_names, 
                                                                categorical_features=categorical_features, categorical_names=categorical_names,
                                                                class_names=class_names, discretize_continuous=False,
                                                                training_labels=self.black_box_labels)                                                            

        # Compute and store variance of each feature
        self.feature_variance = []
        for feature in range(len(self.train_data[0])):
            self.feature_variance.append(np.std(self.train_data[:,feature]))
        # Compute and store the probability of each value for each categorical feature
        self.probability_categorical_feature = []
        if self.categorical_features is not None:
            for nb_feature, feature in enumerate(self.categorical_features):
                set_categorical_value = categorical_values[nb_feature]
                probability_instance_per_feature = []
                for categorical_feature in set_categorical_value:
                    probability_instance_per_feature.append(sum(self.train_data[:,feature] == categorical_feature)/len(self.train_data[:,feature]))
                self.probability_categorical_feature.append(probability_instance_per_feature)
        # We store min, max and mean values of each features in order to generate and evaluate the distance of instances according to the distribution
        self.min_features = []
        self.max_features = []
        self.mean_features = []
        continuous_features = [x for x in set(range(self.train_data.shape[1])).difference(categorical_features)]
        for continuous_feature in continuous_features:
            self.mean_features.append(np.mean(self.train_data[:,continuous_feature]))
            self.max_features.append(max(self.train_data[:,continuous_feature]))
            self.min_features.append(min(self.train_data[:,continuous_feature]))


    def modify_instance_for_linear_model(self, lime_exp, instances_in_field):
        """ 
        Modify the instances in the field to be predict by the trained linear model 
        Args: lime_exp: lime_explainer.explain_instance object
              instances_in_field: Raw values for instances present in the hyper field
        Return: List of instances present in the hyper field in order to be computed by the linear model build by Lime library
        """
        linear_model = lime_exp.easy_model
        used_features = [x for x in lime_exp.used_features]
        if self.categorical_features != []:
            train_enc = instances_in_field[:,self.categorical_features]
            codes = self.enc.transform(train_enc).toarray()
            instances_in_field = np.append(np.asarray(codes), instances_in_field[:,self.continuous_features], axis=1)
        prediction_inside_field = linear_model.predict(instances_in_field[:,used_features])
        return prediction_inside_field


    def transform_data_into_data_frame(self, data_to_transform):
        dictionary = {}
        for nb_feature, name in enumerate(self.feature_names):
            dictionary[name] = data_to_transform[:,nb_feature]
        pandas_frame = pd.DataFrame(dictionary)
        return pandas_frame


    def generate_rule_and_data_for_anchors(self, anchor_exp, target_class, data_to_transform, simulated_user_experiment=False):
        """ 
        Generate rules and data frame for applying anchors 
        Args: anchor_exp: anchor_explainer.explain_instance object
              target_class: Class of the target instance to explain (class predict by the rules from anchor_exp)
              data_to_transform: Train data that are used to generate an anchor
              simulated_user_experiment: Determine if this function returns only the data modify and the rule or the list of features that are used by the anchor
        Return: The rules used by the anchor explanation
                Training data convert to data frame 
        """        
        pandas_frame = self.transform_data_into_data_frame(data_to_transform)

        rules = {}
        features_employed_in_rule = []
        for exp in anchor_exp:
            for feature_number, feature_spliting in enumerate(self.feature_names):
                if "bytes_" in str(type(feature_spliting)):
                    bin_feature_spliting = "b'" + feature_spliting.decode('utf-8') + "'"
                    split = re.split(bin_feature_spliting, exp)
                else:
                    split = re.split(feature_spliting, exp)
                if len(split) > 1:
                    features_employed_in_rule.append(feature_number)
                    signe = str(split[1][:3]).replace(" ", "")
                    comparaison = (split[1][3:]).replace(" ", "")
                    try:
                        comparaison = float(comparaison)
                    # In case the rule is on categorical data and explanation rests on text instead of numerical values
                    except ValueError:
                        for feature_rule in self.categorical_names:
                            if comparaison in self.categorical_names[feature_rule]:
                                comparaison = self.categorical_names[feature_rule].index(comparaison)
                                break
                        if self.verbose: print("Caution ! You're data are not only numbers.")
                    rules[feature_spliting] = [(signe, comparaison, target_class)]
        if simulated_user_experiment:
            return rules, pandas_frame, features_employed_in_rule
        else:
            return rules, pandas_frame
    

    def get_base_model_data(self, set_rules, x_data_frame: pd.DataFrame):
        """
        Filters the trainig data for data points affected by the rules and associated them the prediction of the rule.
        Args: set_rules: A set of rules return by the explanation model
              x_data_fram: a data frame of instances that we want to test whether they validate the set of rules or not
        Return: instances validating the set of rules
        """
        instances_in_anchors = x_data_frame.copy()
        for category, rules in set_rules.items():
            for rule in rules:
                if ">=" in rule[0]:
                    instances_in_anchors = instances_in_anchors.loc[instances_in_anchors[category] >= rule[1]]
                elif "<=" in rule[0]:
                    instances_in_anchors = instances_in_anchors.loc[instances_in_anchors[category] <= rule[1]]
                elif "=" in rule[0]:
                    instances_in_anchors = instances_in_anchors.loc[instances_in_anchors[category] == rule[1]]
                elif "<" in rule[0]:
                    instances_in_anchors = instances_in_anchors.loc[instances_in_anchors[category] < rule[1]]
                elif ">" in rule[0]:
                    instances_in_anchors = instances_in_anchors.loc[instances_in_anchors[category] > rule[1]]
                else:
                    print("Invalid rule detected: {}".format(rule))
        instances_in_anchors = instances_in_anchors.reset_index(drop=True)
        return instances_in_anchors
    
    def generate_artificial_instances_in_anchor(self, instances_in_anchor: pd.DataFrame, nb_instances_in_field, target_instance, 
                                                rules, farthest_distance, growing_method="GF"):
        """
        Generate as many  artificial instances as the number of instances present in the field that validate the anchor rules
        Args: instances_in_anchor: All the instances from the training dataset validatin the anchor rules
              nb_instances_in_field: Number of instances generated in the hyperfield
              target_instance: The target instance to explain
              rules: The set of rules generated by anchor
              farthest_distance: the distance between the target instance and its farthest instance from the training dataset
              method: algorithm to use ('GS' for GrowingSpheres and 'GF' for GrowingFields)
        Return: As many artificial instances as the number of instances present in the field that validate the anchor rules 
        """
        artificial_instances_in_anchors = instances_in_anchor.copy()
        cnt = 2
        while len(artificial_instances_in_anchors) < nb_instances_in_field:
            # If there are not enough instances from the training dataset to compare with instances instances in field 
            # we generate more until we find enough
            # to compare accuracy and coverage of both methods
            try:
                if growing_method == "GS":
                    generated_artificial_instances = generate_inside_ball(target_instance, (0, farthest_distance), int (cnt*nb_instances_in_field))
                if len(self.categorical_features) > 1:
                    if self.verbose: print("farthest distance", farthest_distance)
                    generated_artificial_instances = generate_categoric_inside_ball(target_instance, (0, farthest_distance), 1,
                                                            int (cnt*nb_instances_in_field), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values,
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature,
                                                            min_features=self.min_features, max_features=self.max_features)
                else:
                    if self.verbose: print("farthest distance for not categorical", farthest_distance)
                    generated_artificial_instances = generate_inside_field(target_instance, (0, farthest_distance), 
                                                        int (cnt*nb_instances_in_field), feature_variance=self.feature_variance,
                                                        max_features=self.max_features, min_features=self.min_features)
            except OverflowError:
                    print("over flow error")
            artificial_instances_pandas_frame = self.transform_data_into_data_frame(generated_artificial_instances)
            artificial_instances_in_anchor = self.get_base_model_data(rules, 
                                                                        artificial_instances_pandas_frame)
            artificial_instances_in_anchors = artificial_instances_in_anchors.append(artificial_instances_in_anchor, ignore_index=True)
            cnt += 1
            if cnt > 25 and len(artificial_instances_in_anchors) < 10:
                # If the anchor rule is too specific, it happens that it is impossible to generate instances different from the target instance
                print("anchor rule too specific", rules)
                farthest_distance -= 0.05
                if farthest_distance <= 0.01:
                    # If even wih a very small radius there is not enough instances validating the anchor, we stop
                    return 0
        return artificial_instances_in_anchors[:nb_instances_in_field].to_numpy()

    def store_counterfactual_instances_in_field(self, instances_in_field, target_class=None, libfolding=False):
        """ 
        Store the counterfactual instances present in the field (maximum max_instances counterfactual instances in the field) 
        Args: instances_in_field: Set of instances generated in the hyper field
              target_class: Class of the instances we want to store
              libfolding: Parameter to indicate whether we return the index of the counterfactual present in the field or directly the values
        Return: Depends of libfolding value
        """
        if target_class == None:
            target_class = 1 - self.target_class
        counterfactual_instances_in_field = []
        index_counterfactual_in_field = []
        for index, instance_in_field in enumerate(instances_in_field):
            if self.black_box_predict(instance_in_field.reshape(1, -1)) == target_class:
                counterfactual_instances_in_field.append(instance_in_field)
                index_counterfactual_in_field.append(index)
        return counterfactual_instances_in_field if not libfolding else index_counterfactual_in_field

    def store_instances_in_field_from_class(self, target_class, instances_in_field, instances_in_field_libfolding):
        """
        Store instances sampled in the field from the given target class
        """
        if instances_in_field_libfolding != []:
            # In case of categorical data, we transform categorical values into probability distribution (continuous values for libfolding)
            index_counterfactual_instances_in_field = self.store_counterfactual_instances_in_field(instances_in_field, 
                        target_class=target_class, libfolding=True)
            counterfactual_instances_in_field = instances_in_field[index_counterfactual_instances_in_field]
            counterfactual_libfolding = np.array(instances_in_field_libfolding[index_counterfactual_instances_in_field])
            return counterfactual_instances_in_field, counterfactual_libfolding
        else:
            counterfactual_instances_in_field = np.array(self.store_counterfactual_instances_in_field(instances_in_field, 
                                                    target_class=target_class))
            return counterfactual_instances_in_field, None

    def check_test_unimodal_data(self, instances_in_field, instances_in_field_libfolding=[]):
        """ 
        Test over instances in the hyperfield to discover if data are uni or multimodal
        Args: counterfactual_in_field: Counterfactual instances find in the area of the hyper field
              instances_in_field: All the instances generated or present in the field
              radius: Size of the hyper field
              instances_in_field_libfolding: counterfactual instances with continuous values for Libfolding
        Return: Indicate whether the counterfactual find in the hyper field are unimodal or multimodal 
                and compute the clusters centers in case of multimodal data 
        """    
        # Compute the unimodality test over the set of friends instances
        friends_in_field, friends_libfolding = self.store_instances_in_field_from_class(self.target_class, 
                                                                        instances_in_field, instances_in_field_libfolding)
        friends_results = pf.FTU(friends_libfolding, routine="python") if friends_libfolding is not None \
                    else pf.FTU(friends_in_field, routine="python")

        # Compute the unimodality test over the set of enemies instances
        counterfactual_in_field, counterfactual_libfolding = self.store_instances_in_field_from_class(1-self.target_class, 
                                                                        instances_in_field, instances_in_field_libfolding)
        counterfactual_results = pf.FTU(counterfactual_libfolding, routine="python") if counterfactual_libfolding is not None \
                    else pf.FTU(counterfactual_in_field, routine="python")
            
    
        self.multimodal_friends_results = friends_results.folding_statistics<1
        self.friends_folding_statistics = friends_results.folding_statistics
        self.friends_pvalue = friends_results.p_value

        self.multimodal_counterfactual_results = counterfactual_results.folding_statistics<1
        self.counterfactual_folding_statistics = counterfactual_results.folding_statistics
        self.counterfactual_pvalue = counterfactual_results.p_value
        
        self.separability_index = -1
        self.multimodal_results = (self.multimodal_counterfactual_results or self.multimodal_friends_results)
        
        if self.multimodal_results:
            # If counterfactual instances are multimodal we compute the clusters center 
            # else we test a linear separability problem
            visualizer = KElbowVisualizer(KMeans(), k=(1,8))
            x_elbow = np.array(counterfactual_in_field)
            visualizer.fit(x_elbow)
            n_clusters = visualizer.elbow_value_
            if n_clusters is not None:
                if self.verbose: print("n CLUSTERS ", n_clusters)
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(counterfactual_in_field)
                self.clusters_centers = kmeans.cluster_centers_
                if self.verbose: print("Mean center of clusters from KMEANS ", self.clusters_centers)
        
        # Train a knn to store the 10 closest neighbors to compute the linear separability test
        neigh = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', metric=distances, metric_params={"ape": self})
        labels_in_field = self.black_box_predict(instances_in_field)
        # We ensure that there are as many instances from the target class and the counterfactual class
        target_class_in_field = []
        for label, instance in zip(labels_in_field, instances_in_field):
            if len(target_class_in_field) == len(counterfactual_in_field):
                break
            if label == self.target_class:
                target_class_in_field.append(instance.tolist())
        if len(target_class_in_field) < len(counterfactual_in_field):
            counterfactuals_in_field = counterfactual_in_field[:len(target_class_in_field)]
        else:
            counterfactuals_in_field = counterfactual_in_field
        separability_index_instances = counterfactuals_in_field + target_class_in_field
        random.shuffle(separability_index_instances)
        neigh.fit(separability_index_instances[:1000], self.black_box_predict(separability_index_instances[:1000]))

        # Compute the class of the closest sampled instance for max 500 instances (due to time computation) in the field
        mean = 0
        for item in separability_index_instances[:500]:
            dists, the_result = neigh.kneighbors(item.reshape(1, -1), 10)
            closest_instance = separability_index_instances[the_result[0][9]]
            for nb, dist in enumerate(dists[0]):
                if dist != 0:
                    closest_instance = separability_index_instances[the_result[0][nb]]
                    break
            try:
                if self.black_box_predict(closest_instance.reshape(1, -1)) == self.black_box_predict(item.reshape(1, -1)):
                    mean+=1
            except Exception as inst:
                print(inst)
                print("problem in the search of the closest neighborhood", the_result)     
        mean /= len(separability_index_instances[:500])
        if self.verbose: print("Value of the linear separability test:", mean)
        # We indicate that data are multimodal if the test of linear separability is inferior to the threshold precision
        # of the interpretability methods
        self.separability_index = mean
        
        if not self.multimodal_results:
            self.multimodal_results = mean < self.linear_separability_index
        
        if self.verbose: print("The libfolding test indicates that data are ", "multimodal." if self.multimodal_results else "unimodal.")
        
        if self.categorical_features != [] and (self.counterfactual_pvalue > 0.05 or self.friends_pvalue > 0.05):
            #print("pvalue < 0.05 so we can't trust the multimodality evaluation")
            self.multimodal_results = True
        return True


    def instances_from_dataset_inside_field(self, closest_counterfactual, radius, dataset):
        """
        Counts how many instances from the training data are present in the area of the hyper field
        Args: closest_counterfactual: Center of the hyper field
              radius: Size of the hyper field corresponding to the distance between the target instances and the farthest among the closest counterfactuals
              dataset: Target dataset in order to find instances (between self.train and self.test)
        Return: Index of the instances from the training data that are present in the area of the hyper field
                How many instances from the training data that are in the hyper field
        """
        position_instances_in_field = []
        nb_instance_from_dataset_in_field = 0
        for position, instance_data in enumerate(dataset):
            if distances(closest_counterfactual, instance_data, self) < radius:
                position_instances_in_field.append(position)
                nb_instance_from_dataset_in_field += 1
        if self.verbose: print("nb original instances from the training dataset in the hyperfield : ", nb_instance_from_dataset_in_field)
        # If any true instance are present in the area of the hyperfield, we generate instances based on the percentage of artificial instances
        if nb_instance_from_dataset_in_field == 0 and self.verbose: 
            print("There is any true instances in the area of the hyperfield so we generate based on the percentage of artificial instances in the field.")
        if nb_instance_from_dataset_in_field < self.nb_min_instance_in_field and self.verbose: 
            print("there are not enough instances from the training data in the hyper field so we generate more")
        return position_instances_in_field, nb_instance_from_dataset_in_field 


    def generate_instances_inside_field(self, radius, closest_counterfactual, dataset, farthest_distance, min_instance_per_class,
                                        position_instances_in_field, nb_training_instance_in_field, growing_method="GF", libfolding=False, lime_ls=False):        
        
        """ 
        Generates instances in the  area of the hyper field until minimum instances are found from each class 
        Args: radius: Size of the hyper field
              closest_counterfactual: Counterfactual instance center of the hyper field
              dataset: Target dataset in order to find instances (between self.train and self.test)
              farthest_distance: Distance from the target instance to the farthest training data
              min_instance_per_class: Minimum number of instances from counterfactual class and target class present in the field
              position_instances_in_field: Index of the instances from training data present in the field
              nb_training_instance_in_field: Number of instances from the training data present in the field
              growing_method: The method used to sample artificial instances (between growing spheres and growing fields)
              libfolding: If set to True, compute instances in the area of the hyper field and transform categorical feature into continuous for
                          use of libfolding through distribution values
              lime_ls: Parameter use for experimental evaluation (if the radius of the field is too small we stop generating more instances)
        Return: Set of instances from training data and artificial instances present in the field
                Labels of these instances present in the field
                The percentage of categorical values that are changing depending on the distribution (i.e: radius of the field)
                Set of instances from training data and artificial instances generated for libfolding  
        """
        nb_different_outcome, nb_same_outcome = 0, 0
        generated_instances_inside_field_libfolding = []
        if len(self.categorical_features) > 1 and self.verbose: 
            print("growing field radius", np.round(radius, decimals=3), "percentage of categorical feature changing:", radius/farthest_distance*100)
        start_time = time.time()
        while nb_different_outcome < min_instance_per_class or nb_same_outcome < min_instance_per_class:
            # Compute the percentage of instances generated in the field that have categorical features changing 
            percentage_distribution = radius/farthest_distance*100
            radius = min(1, radius)
            if (time.time() - start_time) > 2:
                # If generating instances is too slow, we add a litle help by increasing the size of the radius
                radius = min(1, radius + 0.005)
                start_time = time.time()
            if ((nb_different_outcome + nb_same_outcome > 10000) and lime_ls):
                break
            # While there is not enough instances from each class
            nb_different_outcome, nb_same_outcome = 0, 0
            try:
                if growing_method == "GS":
                    generated_instances_inside_field = generate_inside_ball(closest_counterfactual, (0, radius), 
                                        max(1, int (self.nb_min_instance_in_field - nb_training_instance_in_field)))
                elif len(self.categorical_features) > 1 and libfolding:
                    # In case of categorical data and computation of categorical feature for libfolding test
                    generated_instances_inside_field, generated_instances_inside_field_libfolding = generate_categoric_inside_ball(closest_counterfactual, 
                                                            (0, radius), percentage_distribution, 
                                                            max(1, int (self.nb_min_instance_in_field - nb_training_instance_in_field)), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                elif len(self.categorical_features) > 1:
                    generated_instances_inside_field = generate_categoric_inside_ball(closest_counterfactual, (0, radius), percentage_distribution,
                                                            max(1, int (self.nb_min_instance_in_field - nb_training_instance_in_field)), 
                                                            self.continuous_features, self.categorical_features, 
                                                            self.categorical_values, feature_variance=self.feature_variance,
                                                            probability_categorical_feature=self.probability_categorical_feature,
                                                            min_features=self.min_features, max_features=self.max_features)
                else:
                    generated_instances_inside_field = generate_inside_field(closest_counterfactual, (0, radius), 
                                                        max(1, int (self.nb_min_instance_in_field - nb_training_instance_in_field)), 
                                                        feature_variance=self.feature_variance, 
                                                        min_features=self.min_features, max_features=self.max_features)
            except OverflowError:
                    print("over flow error")
                    continue
            instances_in_field = np.append(dataset[position_instances_in_field], generated_instances_inside_field, axis=0) if position_instances_in_field != [] else generated_instances_inside_field
            if len(instances_in_field) > len(generated_instances_inside_field_libfolding) and len(self.categorical_features) > 1:
                # If we are dealing with categorical dataset and want to generate more instances for libfolding evaluation
                if libfolding:
                    _, generated_libfolding = generate_categoric_inside_ball(closest_counterfactual, (0, radius), 
                                                            percentage_distribution, len(instances_in_field) - len(generated_instances_inside_field_libfolding), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                else:
                    generated_libfolding = generate_categoric_inside_ball(closest_counterfactual, (0, radius), 
                                                            percentage_distribution, 
                                                            len(instances_in_field) - len(generated_instances_inside_field_libfolding), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                try:
                    generated_instances_inside_field_libfolding = np.append(generated_instances_inside_field_libfolding, generated_libfolding, axis=0)
                except ValueError:
                    generated_instances_inside_field_libfolding = generated_libfolding
            else:
                # In order to not create error with oversampling
                generated_instances_inside_field_libfolding = instances_in_field

            # Compute the number of sampled instances with same and opponent class to the target instance in order to generate more instances 
            # if this is not enough balanced  
            labels_in_field = self.black_box_predict(instances_in_field)
            friends_in_field, ennemies_in_field, friends_in_field_libfolding, ennemies_in_field_libfolding = [], [], [], []
            for label_field, instance_in_field, instance_in_field_libfolding in zip(labels_in_field, instances_in_field, generated_instances_inside_field_libfolding):
                if label_field != self.target_class:
                    ennemies_in_field.append(instance_in_field)
                    ennemies_in_field_libfolding.append(instance_in_field_libfolding)
                    nb_different_outcome += 1
                else:
                    friends_in_field.append(instance_in_field)
                    friends_in_field_libfolding.append(instance_in_field_libfolding)
                    nb_same_outcome += 1
            proportion_same_outcome, proportion_different_outcome = nb_same_outcome/min_instance_per_class, nb_different_outcome/min_instance_per_class
            if proportion_same_outcome < 1 or proportion_different_outcome < 1:
                # data generated inside field are not enough representative so we generate more.
                self.nb_min_instance_in_field += min(proportion_same_outcome, proportion_different_outcome) * min_instance_per_class + min_instance_per_class
        if self.verbose: 
            print('There are ', nb_different_outcome, " instances from a different class in the field over ", len(instances_in_field), " total instances in the dataset.")
            print("There are : ", nb_same_outcome, " instances classified as the target instance in the field.")
        if nb_different_outcome + nb_same_outcome > 10000 and lime_ls:
            return 0

        # We balance the dataset to have almost equal instances from the friend and enemies class
        while len(ennemies_in_field) > 2 * len(friends_in_field):
            friends_in_field = friends_in_field + friends_in_field
            friends_in_field_libfolding = friends_in_field_libfolding + friends_in_field_libfolding
        while len(friends_in_field) > 2 * len(ennemies_in_field):
            ennemies_in_field = ennemies_in_field + ennemies_in_field
            ennemies_in_field_libfolding = ennemies_in_field_libfolding + ennemies_in_field_libfolding
        instances_in_field = friends_in_field + ennemies_in_field
        generated_instances_inside_field_libfolding = friends_in_field_libfolding + ennemies_in_field_libfolding
        random.Random(4).shuffle(instances_in_field)
        random.Random(4).shuffle(generated_instances_inside_field_libfolding)
        instances_in_field = np.array(instances_in_field)
        generated_instances_inside_field_libfolding = np.array(generated_instances_inside_field_libfolding)
        random.shuffle(generated_instances_inside_field_libfolding)

        labels_in_field = self.black_box_predict(instances_in_field)
        return instances_in_field, labels_in_field, generated_instances_inside_field_libfolding

    def compute_linear_regression_accuracy(self, prediction_inside_field, labels_in_field, linear_intercept):
        """
        Function to compute the best threshold for linear regression model and return the accuracy of this model
        Args: prediction_inside_field: Values return by the linear regression explanation model
            labels_in_field: Labels compute by the black box model
        Return: Accuracy of the linear regression explanation model
        """
        accuracys_regression, roc = [], []
        prediction_inside_field_regression, roc_prediction = [], [] 
        for prediction_regression in prediction_inside_field:
            if prediction_regression > linear_intercept:
                prediction_inside_field_regression.append(1-self.target_class)
                roc_prediction.append(1-self.target_class)
            else:
                prediction_inside_field_regression.append(self.target_class)
                roc_prediction.append(self.target_class)
        
        accuracys_regression.append(accuracy_score(prediction_inside_field_regression, labels_in_field))
        try:
            roc.append(roc_auc_score(roc_prediction, labels_in_field))
        except:
            roc.append(0)

        linear_accuracy = max(accuracys_regression)
        linear_auc = max(roc)
        return linear_accuracy, linear_auc

    def compute_labels_inside_field(self, nb_training_instance_in_field, position_instances_in_field, dataset):
        """ 
        computation of the labels for instances in the field 
        Args: nb_training_instance_in_field: Number of instances present in the hyperfield
              position_instances_in_field: Index of the instances from the training data that are in the hyper field
              dataset: Target dataset in order to find instances (between self.train and self.test)
        """
        if nb_training_instance_in_field > 0:
            # Check that there is at least one instance from the training dataset in the area of the hyperfield
            labels_training_instance_in_field = self.black_box_predict(dataset[position_instances_in_field])
            nb_training_instance_in_field_label_as_target = sum(y == self.target_class for y in labels_training_instance_in_field)
            return nb_training_instance_in_field_label_as_target, labels_training_instance_in_field
        else:
            nb_training_instance_in_field_label_as_target = 1
            return nb_training_instance_in_field_label_as_target, None

    def compute_anchor_accuracy_coverage(self, instance, labels_instance_test_data, nb_instances_in_field, 
                                        nb_instance_test_data_label_as_target,
                                        growing_method):
        """
        Computation of Anchors accuracy and coverage
        Args: instance: target instance to explain
              labels_instance_test_data: labels of instances from the testing data
              nb_instances_in_field: Number of instances (artificial and testing) that are in the hyperfield
              nb_instance_test_data_label_as_target: Number of instances from the testing data that are classify as the target instance
              growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
        Return: Anchor's accuracy, Anchors's coverage, Anchors's f2 and Anchor's explanation 
        """
        # In case of multimodal data we generate rule based explanation
        anchor_exp = self.anchor_explainer.explain_instance(instance, self.black_box_predict, threshold=self.threshold_precision, 
                                    delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, 
                                    stop_on_first=False, desired_label=None, beam_size=4)

        # Generate rules and data frame for applying anchors on testing data
        rules, testing_instances_pandas_frame = self.generate_rule_and_data_for_anchors(anchor_exp.names(), self.target_class, self.test_data)
        # Apply anchors and returns instances from testing instances pandas frame with corresponding labels 
        labels_test_instances = self.black_box_predict(self.test_data)
        testing_instances_pandas_frame = pd.concat([testing_instances_pandas_frame, pd.DataFrame(labels_test_instances, columns=["label"])], axis=1)
        testing_instances_in_anchor = self.get_base_model_data(rules, testing_instances_pandas_frame)
        
        # Computes the number of instances from the testing set that are classified as the target instance and validate the anchor rules.
        index_instances_test_data_labels_as_target = np.where([x == self.target_class for x in labels_instance_test_data])
        instances_from_index = self.test_data[index_instances_test_data_labels_as_target]
        testing_instances_in_anchor = testing_instances_in_anchor.drop(columns=['label'])
        coverage_testing_instances_in_anchor = testing_instances_in_anchor.copy()
        nb_test_instances_in_anchor = 0
        for instance_index in instances_from_index:
            matches = coverage_testing_instances_in_anchor[(coverage_testing_instances_in_anchor==instance_index).all(axis=1)]
            if len(matches)>0:
                nb_test_instances_in_anchor += 1
        if nb_test_instances_in_anchor < 10:
            nb_instances_in_field = min(nb_instances_in_field, 100)

        # Generates artificial instances in the area of the anchor rules until there are as many instances as in the hyperfield
        instances_in_anchor = self.generate_artificial_instances_in_anchor(testing_instances_in_anchor, nb_instances_in_field, instance, 
                                                rules, 1, growing_method=growing_method)
        labels_in_anchor = self.black_box_predict(instances_in_anchor)
        anchor_coverage = nb_test_instances_in_anchor/nb_instance_test_data_label_as_target
        
        anchor_accuracy = {'real':None}
        # Compute accuracy over real test instances
        real_labels = []
        for instance_index, label in zip(self.test_data, self.black_box_predict(self.test_data)):
            matches = coverage_testing_instances_in_anchor[(coverage_testing_instances_in_anchor==instance_index).all(axis=1)]
            if len(matches)>0:
                real_labels.append(label)
        
        if real_labels != []:
            anchor_accuracy["real"] = accuracy_score(real_labels, [self.target_class]*len(real_labels))
        
        anchor_accuracy['counterfactual'] = accuracy_score(labels_in_anchor, [self.target_class]*len(labels_in_anchor))
        f2_anchor = (anchor_coverage+2*anchor_accuracy["counterfactual"])/3
        return anchor_accuracy, anchor_coverage, f2_anchor, anchor_exp.names()

    def compute_coverage(self, dataset, instance_center, radius):
        """
        Compute the coverage inside a field
        Args: dataset: Target dataset in order to find instances (between self.train and self.test)
              instance_center: Target instance on which we centered the coverage computation 
              radius: Size of the hyper field
        Return: Coverage of the field centered on instance_center
        """
        borne = []
        for feature, (min_feature, max_feature) in enumerate(zip(self.min_features, self.max_features)):
            # We compute the range of each feature in order to "renormalize" data
            range_feature = max_feature - min_feature
            y = 0
            z = radius * range_feature
            variance = self.feature_variance[feature] * radius
            k = (12 * variance)**0.5
            a1 = min(y, z - k)
            b1 = a1 + k
            borne.append(instance_center[self.continuous_features[feature]] + min(a1, b1, -a1, -b1))
            borne.append(instance_center[self.continuous_features[feature]] + max(a1, b1, -a1, -b1))
        nb_instances_in_borne = 0
        # After calculating the borne values of each feature of the field we check whether each instance is located inside the field or not
        for instance in dataset:
            borne_boolean = True
            for nb, feature_value in enumerate(instance[self.continuous_features]):
                if borne_boolean and feature_value > borne[2*nb] and feature_value < borne[2*nb + 1]:
                    continue
                else:
                    borne_boolean = False
            nb_instances_in_borne += 1 if borne_boolean else 0
        return nb_instances_in_borne/len(dataset)

    def compute_lime_extending_accuracy_coverage(self, train_instances_in_field, instance_at_center_of_field, 
                                                labels_in_field, growing_field, nb_features_employed, farthest_distance, 
                                                growing_method, position_training_instances_in_field, linear_model):
        """ 
        Linear explanation and computation of accuracy inside the initial hyperfield
        Args: train_instances_in_field: Set of training and artificial instances that are present in the hyper field
              instance_at_center_of_field: Target instance on which the explanation is centered 
                    (closest_counterfactual in case of LS and target instance in case of Lime)
              labels_in_field: Corresponding labels predict by the complex model for instance in the hyper field
              growing_field: The growing field object used by APE
              nb_features_employed: Number of features used as explanation by Local Surrogate
              farthest_distance: Distance between the target instance and the farthest instances from the training data
              growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
              position_training_instances_in_field: Index of the instances in the training set that are located in the field
        Return: accuracy, coverage and F1 of the linear surrogate trained over training instances with a logistic regression model
        """
        # We split the artificial instances in train and test set in order to train the linear surrogate over the train and compute 
        # the accuracy over the test data.
        train_instances_in_field, test_instances_in_field, labels_in_field, test_labels_in_field = train_test_split(train_instances_in_field, 
                                                        labels_in_field, test_size=0.4, random_state=42)
        
        # Generate a local surrogate explanation model (centered on the closest counterfactual instance)
        ls_raw_data = self.linear_explainer.explain_instance_training_dataset(instance_at_center_of_field, self.black_box_predict_proba, 
                                                                    model_regressor=linear_model,
                                                                    num_features=nb_features_employed, 
                                                                    instances_in_sphere=train_instances_in_field, 
                                                                    ape=self)
        #print("ls explanation", ls_raw_data.as_list())
        prediction_inside_field = self.modify_instance_for_linear_model(ls_raw_data, test_instances_in_field)

        # Initialize the accuracy of Local Surrogate
        accuracy_ls_raw_data = {'real counterfactual':None, 'real auc':None}
        if position_training_instances_in_field != []:
            real_prediction_inside_field = self.modify_instance_for_linear_model(ls_raw_data, self.train_data[position_training_instances_in_field])
            real_labels_in_field = self.black_box_predict(self.train_data[position_training_instances_in_field])
            accuracy_ls_raw_data["real counterfactual"], accuracy_ls_raw_data["real auc"] = self.compute_linear_regression_accuracy(real_prediction_inside_field,\
                real_labels_in_field, ls_raw_data.easy_model.intercept_)

        accuracy_ls_raw_data["counterfactual"], accuracy_ls_raw_data["auc"] = self.compute_linear_regression_accuracy(prediction_inside_field,\
            test_labels_in_field, ls_raw_data.easy_model.intercept_)
        radius = growing_field.radius
        
        last_radius = radius
        extending = False
        nb_not_increasing, nb_extending = 0, 0
        ini_acuracy_ls_raw_data = accuracy_ls_raw_data
        while radius < 0.5 or nb_extending < 20 and accuracy_ls_raw_data["counterfactual"] > self.threshold_precision:
            # While the radius is not too large or the number of extension without improving the accuracy is less than 20 (too speed up the computation)
            extending = True
            nb_extending += 1
            """ Extending the hyperfield radius until the accuracy inside the hyperfield is lower than the threshold 
            and the radius of the hyper field is not longer than the distances to the farthest instance from the dataset """
            # We first extend the size of the field in order to generate in a bigger space as in GrowingFields
            radius = min(1, radius + 0.05)
            # We generate artificial instances in the field and test if they are new training instance in the field
            position_training_instances_in_field, nb_training_instance_in_field = self.instances_from_dataset_inside_field(instance_at_center_of_field, 
                                                                                                                radius, self.train_data)
            train_instances_in_field, labels_in_field, _ = self.generate_instances_inside_field(radius, instance_at_center_of_field, self.train_data,
                                                                                                                farthest_distance, self.nb_min_instance_per_class_in_field,
                                                                                                                position_training_instances_in_field, nb_training_instance_in_field,
                                                                                                                growing_method=growing_method)

            # We generate artificial instances in the field and test if they are new training instance in the field
            position_testing_instances_in_field, nb_testing_instance_in_field = self.instances_from_dataset_inside_field(instance_at_center_of_field, 
                                                                                                                radius, self.test_data)
            test_instances_in_field, test_labels_in_field, _ = self.generate_instances_inside_field(radius, instance_at_center_of_field, self.test_data,
                                                                                                                farthest_distance, self.nb_min_instance_per_class_in_field,
                                                                                                                position_testing_instances_in_field, nb_testing_instance_in_field,
                                                                                                                growing_method=growing_method)
            
            # Train a new Local Surrogate explanation model on a larger hyper field (with instances inside this hyper field)
            ls_raw_data = self.linear_explainer.explain_instance_training_dataset(instance_at_center_of_field, self.black_box_predict_proba, 
                                                                    num_features=nb_features_employed,
                                                                    model_regressor=linear_model,
                                                                    #self.black_box_predict, model_regressor=model_regressor, 
                                                                    instances_in_sphere=train_instances_in_field,
                                                                    ape=self)
            
            # Compute the accuracy of the new Local Surrogate and replace the accuracy score of the model if it is better than the old one
            prediction_inside_field = self.modify_instance_for_linear_model(ls_raw_data, test_instances_in_field)
            temp_accuracy_ls_raw_data = {'real counterfactual':None, 'real auc':None}
            if position_training_instances_in_field != []:
                real_temp_prediction_inside_field = self.modify_instance_for_linear_model(ls_raw_data, self.train_data[position_training_instances_in_field])
                real_temp_labels_in_field = self.black_box_predict(self.train_data[position_training_instances_in_field])
                temp_accuracy_ls_raw_data["real counterfactual"], temp_accuracy_ls_raw_data["real auc"] = \
                    self.compute_linear_regression_accuracy(real_temp_prediction_inside_field, \
                        real_temp_labels_in_field, ls_raw_data.easy_model.intercept_)
            temp_accuracy_ls_raw_data["counterfactual"], temp_accuracy_ls_raw_data["auc"] = \
                self.compute_linear_regression_accuracy(prediction_inside_field, test_labels_in_field,\
                    ls_raw_data.easy_model.intercept_)
            if accuracy_ls_raw_data["counterfactual"] <= temp_accuracy_ls_raw_data["counterfactual"]: 
                accuracy_ls_raw_data = temp_accuracy_ls_raw_data
                #print("increasing accuracy")
                final_accuracy = accuracy_ls_raw_data
                last_radius = radius
                nb_not_increasing = 0
            else:
                final_accuracy = accuracy_ls_raw_data
                #print("keeping accuracy")
                if nb_not_increasing == 0:
                    last_radius -= 0.005
                nb_not_increasing += 1
                if nb_not_increasing == 10:
                    # If the accuracy is not increasing after 10 times we increase the size of the radius, then we stop to extend the hyperfield
                    break

        if extending:
            accuracy_ls_raw_data = final_accuracy
            radius = last_radius
        
        test_data_target_label = self.test_data[np.where([x == self.target_class for x in self.black_box_predict(self.test_data)])]
        lime_extending_coverage = self.compute_coverage(test_data_target_label, instance_at_center_of_field, radius)
        f2_lime_extending = (2*accuracy_ls_raw_data['counterfactual'] + lime_extending_coverage)/3
        return accuracy_ls_raw_data, ini_acuracy_ls_raw_data, lime_extending_coverage,  f2_lime_extending, ls_raw_data.as_list(), radius

    def compute_decision_tree_accuracy_coverage(self, train_instances_in_field, labels_in_field, instance_at_center_of_field, radius, 
                                                position_training_instance):
        """ 
        Shallow Decision Tree explanation and computation of accuracy inside the initial hyperfield
        Args: train_instances_in_field: Set of training and artificial instances that are present in the hyper field
              labels_in_field: Corresponding labels predict by the complex model for instance in the hyper field
              instance_at_center_of_field: Target instance on which the explanation is centered 
                    (closest_counterfactual in case of LS and target instance in case of Lime)
              growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
              position_training_instances_in_field: Index of the instances in the training set that are located in the field
        Return: accuracy, coverage and F1 of the shallow Decision Tree trained over training instances in a field centered on the 
                closest counterfactual with radius equals to the distance between target instance and closest counterfactual 
        """
        # We split the artificial instances in train and test set in order to train the Decision Tree surrogate over the train and compute 
        # the accuracy over the test data.
        train_instances_in_field, test_instances_in_field, labels_in_field, test_labels_in_field = train_test_split(train_instances_in_field, 
                                                        labels_in_field, test_size=0.4, random_state=42)
        
        # Generate a local surrogate explanation model (centered on the closest counterfactual instance) trained over 
        # training instances with a Decision Tree model as explanation
        decision_tree = tree.DecisionTreeClassifier(max_depth=2)
        decision_tree.fit(train_instances_in_field, labels_in_field)
        prediction_inside_field = decision_tree.predict(test_instances_in_field)
        # Initialize the accuracy of the Decision Tree
        accuracy_decision_tree = {'real':None}
        if position_training_instance != []:
            real_labels_in_fields = self.black_box_predict(self.train_data[position_training_instance])
            real_prediction_inside_fields = decision_tree.predict(self.train_data[position_training_instance])
            accuracy_decision_tree['real'] = accuracy_score(real_labels_in_fields, real_prediction_inside_fields)
        accuracy_decision_tree['counterfactual'] = accuracy_score(test_labels_in_field, prediction_inside_field)
        
        """ computation of the coverage inside the field for linear model on training data """
        test_data_target_label = self.test_data[np.where([x == self.target_class for x in self.black_box_predict(self.test_data)])]
        decision_tree_coverage = self.compute_coverage(test_data_target_label, instance_at_center_of_field, radius)
        f2_decision_tree = (2*accuracy_decision_tree['counterfactual'] + decision_tree_coverage)/3
        return accuracy_decision_tree, decision_tree_coverage, f2_decision_tree, decision_tree.tree_, radius

    def explain_instance(self, instance, opponent_class=None, growing_method='GF', n_instance_per_layer=2000, first_radius=0.01, 
                        nb_features_employed=None, dicrease_radius=10, linear_model=None, all_explanations_model=False, user_experiments=False, 
                        k_closest=False, time_k_closest=False, distance_metric="w_euclidian"):
        """
        Returns either an explanation from anchors or lime along with one or multiple counter factual explanation
        Args: instance: Target instance to explain
              opponent_class: Class of the desired counterfactual instance
              growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
              n_instance_per_layer: Hyperparameter of the number of instances we want to generate for growing_method
              first_radius: Hyperparameter to initalize the radius of the growing_method
              nb_features_employed: Indicate how many features will be used as explanation for the linear explanation (used also for experiments)
              dicrease_radius: Hyperparameter to indicate the speed to dicrease the radius of the growing_method
              all_explanations_model: generate explanation with multiple explanation models (for experiments)
              user_experiments: return features employed by linear and rule based explanation (for experiments)
              k_closest: compute the average distance to the k closest instances from the testing set to either the closest counterfactual (unimodal case)
                        or to each center of cluster (mulimodal case) 
        Return: APE's coverage
                APE's accuracy
                APE's F1
                Indicate whether counter factual instances are multimodal: 1 or unimodal: 0
        """
        self.farthest_distance = None
        nb_features_employed = len(instance) if nb_features_employed == None else nb_features_employed
        self.target_class = self.black_box_predict(instance.reshape(1, -1))[0]
        # Computes the distance to the farthest instance from the training dataset to bound generating instances 
        farthest_distance = 0
        for training_instance in self.train_data:
            farthest_distance_now = distances(training_instance, instance, self, metrics=distance_metric)
            if farthest_distance_now > farthest_distance:
                farthest_distance = farthest_distance_now

        # The growing method (GS or GF) is searching for the closest counterfactual
        growing_field = cf.CounterfactualExplanation(instance, self.black_box_predict, method=growing_method, target_class=opponent_class, 
                                                    continuous_features=self.continuous_features, 
                                                    categorical_features=self.categorical_features, 
                                                    categorical_values=self.categorical_values,
                                                    max_features=self.max_features,
                                                    min_features=self.min_features)
        growing_field.fit(n_in_layer=n_instance_per_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=True, 
                                                    verbose=self.verbose, feature_variance=self.feature_variance, 
                                                    farthest_distance_training_dataset=farthest_distance, 
                                                    probability_categorical_feature=self.probability_categorical_feature, 
                                                    min_counterfactual_in_sphere=self.nb_min_instance_per_class_in_field)
        self.closest_counterfactual = growing_field.enemy

        # Computes the distance to the farthest instance from the training dataset to bound generating instances 
        farthest_distance_cf = 0
        for training_instance in self.train_data:
            farthest_distance_cf_now = distances(self.closest_counterfactual, training_instance, self, metrics=distance_metric)
            if farthest_distance_cf_now > farthest_distance_cf:
                farthest_distance_cf = farthest_distance_cf_now
        self.farthest_distance = farthest_distance_cf

        # For experiments results we can either compute the average distance to the k closest experiments or indicate 
        # that the growing method discovered the closest counterfactual
        if k_closest is not False:
            return k_closest_experiments(self, k_closest)
        elif time_k_closest is not False:
            return True
        
        min_instance_per_class = self.nb_min_instance_per_class_in_field
        # Generates or store training instances in the area of the hyperfield and their corresponding labels
        position_training_instances_in_field, nb_training_instance_in_field = self.instances_from_dataset_inside_field(self.closest_counterfactual, 
                                                                                                    growing_field.radius, self.train_data)
        training_instances_in_field, train_labels_in_field, instances_in_field_libfolding = self.generate_instances_inside_field(growing_field.radius, 
                                                                                                    self.closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                    min_instance_per_class, position_training_instances_in_field, 
                                                                                                    nb_training_instance_in_field, libfolding=True,
                                                                                                                growing_method=growing_method)

        # Generates or store testing instances in the area of the hyperfield and their corresponding labels
        position_testing_instances_in_field, nb_testing_instance_in_field = self.instances_from_dataset_inside_field(self.closest_counterfactual, 
                                                                                                                growing_field.radius, self.test_data)
        test_instances_in_field, test_labels_in_field, _ = self.generate_instances_inside_field(growing_field.radius, 
                                                                                                    self.closest_counterfactual, 
                                                                                                    self.test_data,
                                                                                                    farthest_distance_cf, 
                                                                                                    self.nb_min_instance_per_class_in_field,
                                                                                                    position_testing_instances_in_field, 
                                                                                                    nb_testing_instance_in_field,
                                                                                                                growing_method=growing_method)
        
        # Compute the libfolding test to verify wheter instances in the area of the hyper field is multimodal or unimodal
        unimodal_test = self.check_test_unimodal_data(training_instances_in_field, instances_in_field_libfolding)
        nb = 0
        while not unimodal_test:
            # While the libfolding test is not able to declare that data are multimodal or unimodal we extend the number of instances that are generated
            min_instance_per_class *= 1.5
            training_instances_in_field, train_labels_in_field, instances_in_field_libfolding = self.generate_instances_inside_field(growing_field.radius, 
                                                                                                    self.closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                    min_instance_per_class, position_training_instances_in_field, 
                                                                                                    nb_training_instance_in_field, libfolding=True,
                                                                                                                growing_method=growing_method)

            unimodal_test = self.check_test_unimodal_data(training_instances_in_field, instances_in_field_libfolding)        
            if self.verbose:
                print("nb times libfolding is not able to determine wheter datas are unimodal or multimodal:", nb)
                print()
            nb += 1

        """ Different cases for experiments """
        if user_experiments:
            return simulate_user_experiments(self, instance, nb_features_employed, farthest_distance_cf, 
                                            self.closest_counterfactual, growing_field,
                                            position_training_instances_in_field, nb_training_instance_in_field)

        # Computes the labels for instances from the dataset to compute accuracy for explanation method 
        labels_instance_test_data = self.black_box_predict(self.test_data)
        nb_instance_test_data_label_as_target = sum(x == self.target_class for x in labels_instance_test_data)

        if all_explanations_model:
            return compute_all_explanation_method_accuracy(self, instance, growing_field, nb_instance_test_data_label_as_target,
                                            position_training_instances_in_field, training_instances_in_field, train_labels_in_field,
                                            farthest_distance_cf, nb_features_employed,
                                            growing_method, linear_model)

        # If there is no experiments, we return either a linear or a rule based explanation
        if self.multimodal_results:
            # In case of multimodal data, we generate a rule based explanation and compute accuracy and coverage of this explanation model
            ape_accuracy, ape_coverage, ape_f2, ape_explanation = self.compute_anchor_accuracy_coverage(instance, 
                                        labels_instance_test_data, len(test_instances_in_field), 
                                        nb_instance_test_data_label_as_target,
                                        growing_method=growing_method)
        
        else:
            # In case of unimodal data, we generate linear explanation and compute accuracy and coverage of this explanation model
            ape_accuracy, ape_accuracy_not_extending, ape_coverage, ape_f2, ape_explanation, self.extended_radius = self.compute_lime_extending_accuracy_coverage(training_instances_in_field, 
                                        self.closest_counterfactual, train_labels_in_field, growing_field, nb_features_employed,
                                        farthest_distance_cf, growing_method, position_training_instances_in_field, linear_model)

        return ape_coverage, ape_accuracy, ape_f2, ape_explanation, 1 if self.multimodal_results else 0
