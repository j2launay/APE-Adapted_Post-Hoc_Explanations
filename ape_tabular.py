import copy
import re
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from yellowbrick.cluster import KElbowVisualizer
from anchors import utils, anchor_tabular, anchor_base, limes
from anchors.limes.utils_stability import compute_WLS_stdevs, refactor_confints_todict, compare_confints
from growingspheres import counterfactuals as cf
from growingspheres.utils.gs_utils import generate_inside_field, generate_categoric_inside_ball, distances, get_distances
from ape_tabular_experiments import compute_all_explanation_method_precision, simulate_user_experiments, compute_local_surrogate_precision_coverage, ape_illustrative_results, simulate_user_experiments_lime_ls
import pyfolding as pf
from sklearn.metrics import precision_score
from sklearn.neighbors import NearestNeighbors
import time

class ApeTabularExplainer(object):
    """
    Args:

    """
    def __init__(self, train_data, class_names, black_box_predict, black_box_predict_proba=None,
                multiclass = False, continuous_features=None, categorical_features=None,
                categorical_values = None, feature_names=None, discretizer="MDLP", 
                nb_min_instance_in_sphere=800, threshold_precision=0.95, 
                nb_min_instance_per_class_in_sphere=100, verbose=False, 
                categorical_names=None, linear_separability_index=0.99):
        train_data, test_data = train_test_split(train_data, test_size=0.4, random_state=42)
        self.train_data = train_data
        self.test_data = test_data
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
        self.feature_names= feature_names
        self.discretizer = discretizer
        self.nb_min_instance_in_sphere = nb_min_instance_in_sphere
        self.threshold_precision = threshold_precision
        self.nb_min_instance_per_class_in_sphere = nb_min_instance_per_class_in_sphere
        self.verbose = verbose
        self.linear_separability_index = linear_separability_index
        self.black_box_labels = black_box_predict(self.train_data)
        if self.verbose: print("Setting interpretability methods")
        self.anchor_explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, train_data, 
                                                                    copy.copy(categorical_names), discretizer=discretizer, 
                                                                    black_box_labels=self.black_box_labels, ordinal_features=continuous_features)
        self.lime_explainer = limes.lime_tabular.LimeTabularExplainer(train_data, feature_names=feature_names, 
                                                                categorical_features=categorical_features, categorical_names=categorical_names,
                                                                class_names=class_names, discretize_continuous=True, discretizer=discretizer, 
                                                                training_labels=self.black_box_labels)                                                            
        # Compute and store variance of each feature
        self.feature_variance = []
        for feature in range(len(train_data[0])):
            self.feature_variance.append(np.var(train_data[:,feature]))
        # Compute and store the probability of each value for each categorical feature
        self.probability_categorical_feature = []
        if self.categorical_features is not None:
            for nb_feature, feature in enumerate(self.categorical_features):
                set_categorical_value = categorical_values[nb_feature]
                probability_instance_per_feature = []
                for categorical_feature in set_categorical_value:
                    probability_instance_per_feature.append(sum(self.train_data[:,feature] == categorical_feature)/len(self.train_data[:,feature]))
                self.probability_categorical_feature.append(probability_instance_per_feature)
        self.min_features = []
        self.max_features = []
        continuous_features = [x for x in set(range(train_data.shape[1])).difference(categorical_features)]
        for continuous_feature in continuous_features:
            self.max_features.append(max(train_data[:,continuous_feature]))
            self.min_features.append(min(train_data[:,continuous_feature]))


    def modify_instance_for_linear_model(self, lime_exp, instances_in_sphere):
        """ 
        Modify the instances in the sphere to be predict by the linear model trained in Lime 
        Args: lime_exp: lime_explainer.explain_instance object
              instances_in_sphere: Raw values for instances present in the hyper field
        Return: List of instances present in the hyper field in order to be computed by the linear model build by Lime
        """
        linear_model = lime_exp.easy_model
        used_features = [x for x in lime_exp.used_features]
        prediction_inside_sphere = linear_model.predict(instances_in_sphere[:,used_features])
        return prediction_inside_sphere


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
    
    def generate_artificial_instances_in_anchor(self, instances_in_anchor: pd.DataFrame, nb_instances_in_sphere, target_instance, 
                                                rules, farthest_distance, percentage_distribution):
        """
        Generate as many  artificial instances as the number of instances present in the field that validate the anchor rules
        Args: instances_in_anchor: All the instances from the training dataset validatin the anchor rules
              nb_instances_in_sphere: Number of instances generated in the hyperfield
              target_instance: The target instance to explain
              rules: The set of rules generated by anchor
              farthest_distance: the distance between the target instance and its farthest instance from the training dataset
              percentage_distribution: Size of the hyper field for categorical data
        Return: As many artificial instances as the number of instances present in the field that validate the anchor rules 
        """
        artificial_instances_in_anchors = instances_in_anchor.copy()
        cnt = 2
        while len(artificial_instances_in_anchors) < nb_instances_in_sphere:
            # If there are not enough instances from the training dataset to compare with instances instances in sphere we generate more until we find enough
            # to compare precision and coverage of both methods
            try:
                if len(self.categorical_features) > 1:
                    generated_artificial_instances = generate_categoric_inside_ball(target_instance, (0, farthest_distance), 1,
                                                            int (cnt*nb_instances_in_sphere), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values,
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature,
                                                            min_features=self.min_features, max_features=self.max_features)
                else:
                    generated_artificial_instances = generate_inside_field(target_instance, (0, farthest_distance), 
                                                        int (cnt*nb_instances_in_sphere), feature_variance=self.feature_variance,
                                                        max_features=self.max_features, min_features=self.min_features)
                                                        #max_features=self.max_features, min_features=self.min_features)
            except OverflowError:
                    print("over flow error")
            artificial_instances_pandas_frame = self.transform_data_into_data_frame(generated_artificial_instances)
            artificial_instances_in_anchor = self.get_base_model_data(rules, 
                                                                        artificial_instances_pandas_frame)
            artificial_instances_in_anchors = artificial_instances_in_anchors.append(artificial_instances_in_anchor, ignore_index=True)
            cnt += 1
        return artificial_instances_in_anchors[:nb_instances_in_sphere].to_numpy()

    def store_counterfactual_instances_in_sphere(self, instances_in_sphere, target_class, libfolding=False):
        """ 
        Store the counterfactual instances present in the sphere (maximum max_instances counterfactual instances in the sphere) 
        Args: instances_in_sphere: Set of instances generated in the hyper field
              target_class: Class of the target instance
              libfolding: Parameter to indicate whether we return the index of the counterfactual present in the field or directly the values
        Return: Depends of libfolding value
        """
        counterfactual_instances_in_sphere = []
        index_counterfactual_in_sphere = []
        for index, instance_in_sphere in enumerate(instances_in_sphere):
            if self.black_box_predict(instance_in_sphere.reshape(1, -1)) != self.target_class:
                counterfactual_instances_in_sphere.append(instance_in_sphere)
                index_counterfactual_in_sphere.append(index)
        return counterfactual_instances_in_sphere if not libfolding else index_counterfactual_in_sphere

    def check_test_unimodal_data(self, counterfactual_in_sphere, instances_in_sphere, radius, counterfactual_libfolding=None):
        """ 
        Test over instances in the hypersphere to discover if data are uni or multimodal
        Args: counterfactual_in_sphere: Counterfactual instances find in the area of the hyper field
              instances_in_sphere: All the instances generated or present in the field
              radius: Size of the hyper field
              counterfactual_libfolding: counterfactual instances with continuous values for Libfolding
        Return: Indicate whether the counterfactual find in the hyper field are unimodal or multimodal 
                and compute the clusters centers in case of multimodal data 
        """
        #try:
        start_time = time.time()
        results = pf.FTU(counterfactual_libfolding, routine="python") if counterfactual_libfolding is not None else pf.FTU(counterfactual_in_sphere, routine="python")
        #print("--- %s seconds --- to compute the unimodality test" % (time.time() - start_time))
        start_time = time.time()
        self.multimodal_results = results.folding_statistics<1
        #print(self.multimodal_results)
        if self.multimodal_results:
            # If counterfactual instances are multimodal we compute the clusters center 
            visualizer = KElbowVisualizer(KMeans(), k=(1,8))
            x_elbow = np.array(counterfactual_in_sphere)
            visualizer.fit(x_elbow)
            n_clusters = visualizer.elbow_value_
            if n_clusters is not None:
                if self.verbose: print("n CLUSTERS ", n_clusters)
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(counterfactual_in_sphere)
                self.clusters_centers = kmeans.cluster_centers_
                if self.verbose: print("Mean center of clusters from KMEANS ", self.clusters_centers)
          #print("--- %s seconds --- to compute clusters center" % (time.time() - start_time))
            #start_time = time.time()
        else:
            # If counterfactual instances are unimodal we test a linear separability problem
          #print("nb instances in hyper field", len(instances_in_sphere))
            start_time = time.time()
            neigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=distances, metric_params={"ape": self})
            neigh.fit(instances_in_sphere, self.black_box_predict(instances_in_sphere))
          #print("--- %s seconds to fit the KNN ---" % (time.time() - start_time))
            start_time = time.time()
            #tree_closest_neighborhood = scipy.spatial.cKDTree(instances_in_sphere)
            #print("--- %s seconds ---" % (time.time() - start_time))

            mean = 0
            target_class = self.black_box_predict(counterfactual_in_sphere[0].reshape(1, -1)) 
          #print("counterfactual in the hyper field", len(counterfactual_in_sphere))
            for item in counterfactual_in_sphere[:500]:
                #the_result = tree_closest_neighborhood.query(item, k=2)
                #print("the result tree", the_result)
                _, the_result = neigh.kneighbors(item.reshape(1, -1), 2)
                try:
                    if self.black_box_predict(instances_in_sphere[the_result[0][1]].reshape(1, -1)) == target_class:
                        mean+=1
                except Exception as inst:
                    print(inst)
                    print("problem in the search of the closest neighborhood", the_result)     
            mean /= len(counterfactual_in_sphere[:500])
          #print("--- %s seconds to compute the linear separability test ---" % (time.time() - start_time))
            start_time = time.time()
            if self.verbose: print("Value of the linear separability test:", mean)
            # We indicate that data are multimodal if the test of linear separability is inferior to the threshold precision
            # of the interpretability methods 
            self.multimodal_results = mean < self.linear_separability_index 
          #print("multimodal results:", self.multimodal_results)
        if self.verbose: print("The libfolding test indicates that data are ", "multimodal." if self.multimodal_results else "unimodal.")
        return True


    def instances_from_dataset_inside_sphere(self, closest_counterfactual, radius, dataset):
        """
        Counts how many instances from the training data are present in the area of the hyper field
        Args: closest_counterfactual: Center of the hyper field
              radius: Size of the hyper field corresponding to the distance between the target instances and the farthest among the closest counterfactuals
        Return: Index of the instances from the training data that are present in the area of the hyper field
                How many instances from the training data that are in the hyper field
        """
        position_instances_in_sphere = []
        nb_instance_from_dataset_in_sphere = 0
        for position, instance_data in enumerate(dataset):
            #if get_distances(closest_counterfactual, instance_data, categorical_features=self.categorical_features)["euclidean"] < radius:
            #x, y, dataset, max_feature, min_feature, categorical_features=[]
            if distances(closest_counterfactual, instance_data, self) < radius:
                position_instances_in_sphere.append(position)
                nb_instance_from_dataset_in_sphere += 1
        if self.verbose: print("nb original instances from the training dataset in the hypersphere : ", nb_instance_from_dataset_in_sphere)
        # If any true instance are present in the area of the hypersphere, we generate instances based on the percentage of artificial instances
        if nb_instance_from_dataset_in_sphere == 0 and self.verbose: 
            print("There is any true instances in the area of the hypersphere so we generate based on the percentage of artificial instances in the sphere.")
        if nb_instance_from_dataset_in_sphere < self.nb_min_instance_in_sphere and self.verbose: 
            print("there are not enough instances from the training data in the hyper sphere so we generate more")
        return position_instances_in_sphere, nb_instance_from_dataset_in_sphere 


    def generate_instances_inside_sphere(self, radius, closest_counterfactual, dataset, farthest_distance, min_instance_per_class,
                                        position_instances_in_sphere, nb_training_instance_in_sphere, libfolding=False, lime_ls=False):        
        
        """ 
        Generates instances in the  area of the hyper field until minimum instances are found from each class 
        Args: radius: Size of the hyper field
              closest_counterfactual: Counterfactual instance center of the hyper field
              farthest_distance: Distance from the target instance to the farthest training data
              min_instance_per_class: Minimum number of instances from counterfactual class and target class present in the field
              position_instances_in_sphere: Index of the instances from training data present in the field
              nb_training_instance_in_sphere: Number of instances from the training data present in the field
              libfolding: If set to True, compute instances in the area of the hyper field and transform categorical feature into continuous for
                          use of libfolding through distribution values
        Return: Set of instances from training data and artificial instances present in the field
                Labels of these instances present in the field
                The percentage of categorical values that are changing depending on the distribution (i.e: radius of the field)
                Set of instances from training data and artificial instances generated for libfolding  
        """
        nb_different_outcome, nb_same_outcome, iteration = 0, 0, 0
        generated_instances_inside_sphere_libfolding = []
        # Compute the percentage of instances generated in the sphere that have categorical features changing 
        percentage_distribution = radius/farthest_distance*100
        if len(self.categorical_features) > 1 and self.verbose: 
            print("growing sphere radius", np.round(radius, decimals=3), "percentage of categorical feature changing:", percentage_distribution)
        start_time = time.time()
        while nb_different_outcome < min_instance_per_class or nb_same_outcome < min_instance_per_class:
            percentage_distribution = radius/farthest_distance*100
            if (time.time() - start_time) > 2:
              #print("--- %s seconds ---" % (time.time() - start_time))
              #print("nb same outcome", nb_same_outcome)
              #print("nb different outcome", nb_different_outcome)
              #print("radius", radius)
              #print("libfolding", libfolding)
              #print("percentage cateforcia", percentage_distribution) 
                radius += 0.005
                start_time = time.time()
            if ((nb_different_outcome + nb_same_outcome > 10000) and lime_ls):
                break
            # While there is not enough instances from each class
            nb_different_outcome, nb_same_outcome = 0, 0
            try:
                if len(self.categorical_features) > 1 and libfolding:
                    # In case of categorical data and computation of categorical feature for libfolding test
                    generated_instances_inside_sphere, generated_instances_inside_sphere_libfolding = generate_categoric_inside_ball(closest_counterfactual, 
                                                            (0, radius), percentage_distribution, 
                                                            max(1, int (self.nb_min_instance_in_sphere - nb_training_instance_in_sphere)), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                elif len(self.categorical_features) > 1:
                    generated_instances_inside_sphere = generate_categoric_inside_ball(closest_counterfactual, (0, radius), percentage_distribution,
                                                            max(1, int (self.nb_min_instance_in_sphere - nb_training_instance_in_sphere)), 
                                                            self.continuous_features, self.categorical_features, 
                                                            self.categorical_values, feature_variance=self.feature_variance,
                                                            probability_categorical_feature=self.probability_categorical_feature,
                                                            min_features=self.min_features, max_features=self.max_features)
                else:
                    generated_instances_inside_sphere = generate_inside_field(closest_counterfactual, (0, radius), 
                                                        max(1, int (self.nb_min_instance_in_sphere - nb_training_instance_in_sphere)), 
                                                        feature_variance=self.feature_variance, 
                                                        min_features=self.min_features, max_features=self.max_features)
            except OverflowError:
                    print("over flow error")

            instances_in_sphere = np.append(dataset[position_instances_in_sphere], generated_instances_inside_sphere, axis=0) if position_instances_in_sphere != [] else generated_instances_inside_sphere
            if len(instances_in_sphere) > len(generated_instances_inside_sphere_libfolding) and len(self.categorical_features) > 1:
                if libfolding:
                    _, generated_libfolding = generate_categoric_inside_ball(closest_counterfactual, (0, radius), 
                                                            percentage_distribution, len(instances_in_sphere) - len(generated_instances_inside_sphere_libfolding), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                else:
                    generated_libfolding = generate_categoric_inside_ball(closest_counterfactual, (0, radius), 
                                                            percentage_distribution, 
                                                            len(instances_in_sphere) - len(generated_instances_inside_sphere_libfolding), 
                                                            self.continuous_features, self.categorical_features, self.categorical_values, 
                                                            feature_variance=self.feature_variance, 
                                                            probability_categorical_feature=self.probability_categorical_feature, 
                                                            libfolding=libfolding, min_features=self.min_features,
                                                            max_features=self.max_features)
                try:
                    generated_instances_inside_sphere_libfolding = np.append(generated_instances_inside_sphere_libfolding, generated_libfolding, axis=0)
                except ValueError:
                    generated_instances_inside_sphere_libfolding = generated_libfolding
            labels_in_sphere = self.black_box_predict(instances_in_sphere)
            for label_sphere in labels_in_sphere:
                if label_sphere != self.target_class:
                    nb_different_outcome += 1
                else:
                    nb_same_outcome += 1
            proportion_same_outcome, proportion_different_outcome = nb_same_outcome/min_instance_per_class, nb_different_outcome/min_instance_per_class
            if proportion_same_outcome < 1 or proportion_different_outcome < 1:
                # data generated inside sphere are not enough representative so we generate more.
                self.nb_min_instance_in_sphere += min(proportion_same_outcome, proportion_different_outcome) * min_instance_per_class + min_instance_per_class
                
        if self.verbose: 
            print('There are ', nb_different_outcome, " instances from a different class in the sphere over ", len(instances_in_sphere), " total instances in the dataset.")
            print("There are : ", nb_same_outcome, " instances classified as the target instance in the sphere.")
        if nb_different_outcome + nb_same_outcome > 10000 and lime_ls:
            return 0
        return instances_in_sphere, labels_in_sphere, percentage_distribution, generated_instances_inside_sphere_libfolding

    def compute_linear_regression_precision(self, prediction_inside_sphere, labels_in_sphere):
        """
        Function to compute the best threshold for linear regression model and return the precision of this model
        Args: prediction_inside_sphere: Values return by the linear regression explanation model
            labels_in_sphere: Labels compute by the black box model
        Return: Precision of the linear regression explanation model
        """
        # Store the minimum and maximum prediction values as baseline for the regression threshold
        min_threshold_regression, max_threshold_regression = min(prediction_inside_sphere), max(prediction_inside_sphere)
        try:
            # Set 10 threshold values for the regression model between the min and the max
            thresholds_regression = np.arange(min_threshold_regression, max_threshold_regression, (max_threshold_regression-min_threshold_regression)/10 )
        except ValueError:
            thresholds_regression = [min_threshold_regression, max_threshold_regression]
        precisions_regression = []
        for threshold_regression in thresholds_regression:
            prediction_inside_sphere_regression_test = []
            for prediction_regression in prediction_inside_sphere:
                # TODO regarder si c'est toujours la classe 1 quand c'est supérieur et 0 inférieur + S'occuper des cas multiclasses
                if prediction_regression > threshold_regression:
                    prediction_inside_sphere_regression_test.append(1)
                else:
                    prediction_inside_sphere_regression_test.append(0)
            #precision_regression = sum(prediction_inside_sphere_regression_test == labels_in_sphere)/len(prediction_inside_sphere_regression_test)
            #print("precision regression", precision_regression)
            precision_regression = precision_score(prediction_inside_sphere_regression_test, labels_in_sphere)
            #print("average precision", average_precision)
            precisions_regression.append(precision_regression)
        lime_extending_precision = max(precisions_regression)
        return lime_extending_precision

    def compute_labels_inside_sphere(self, nb_training_instance_in_sphere, position_instances_in_sphere, dataset):
        """ 
        computation of the labels for instances in the field 
        Args: nb_training_instance_in_sphere: Number of instances present in the hyperfield
              position_instances_in_sphere: Index of the instances from the training data that are in the hyper field
        """
        if nb_training_instance_in_sphere > 0:
            # Check that there is at least one instance from the training dataset in the area of the hypersphere
            labels_training_instance_in_sphere = self.black_box_predict(dataset[position_instances_in_sphere])
            nb_training_instance_in_sphere_label_as_target = sum(y == self.target_class for y in labels_training_instance_in_sphere)
            return nb_training_instance_in_sphere_label_as_target, labels_training_instance_in_sphere
        else:
            nb_training_instance_in_sphere_label_as_target = 1
            return nb_training_instance_in_sphere_label_as_target, None

    def compute_anchor_precision_coverage(self, instance, labels_instance_test_data, nb_instances_in_sphere, 
                                        farthest_distance, percentage_distribution, nb_instance_test_data_label_as_target):
        """
        Computation of Anchors precision and coverage
        Args: instance: target instance to explain
              labels_instance_train_data: labels of instances from the training data
              nb_instances_in_sphere: Number of instances (artificial and training) that are in the hyperfield
              farthest_distance: distance from the target instance to the farthest instance from the training data
              percentage_distribution: Percentage of instances whose categorical values will be change (radius of the field for categorical data)
              nb_instance_train_data_label_as_target: Number of instances from the training data that are classify as the target instance
        Return: Anchor's precision, Anchors's coverage, Anchors's f2 and Anchor's explanation 
        """
        # In case of multimodal data we generate rule based explanation
        anchor_exp = self.anchor_explainer.explain_instance(instance, self.black_box_predict, threshold=self.threshold_precision, 
                                    delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, 
                                    stop_on_first=False, desired_label=None, beam_size=4)
        # Generate rules and data frame for applying anchors on training data

        rules, testing_instances_pandas_frame = self.generate_rule_and_data_for_anchors(anchor_exp.names(), self.target_class, self.test_data)
        # Apply anchors and returns instances from testing instances pandas frame with corresponding labels 
        labels_test_instances = self.black_box_predict(self.test_data)
        testing_instances_pandas_frame = pd.concat([testing_instances_pandas_frame, pd.DataFrame(labels_test_instances, columns=["label"])], axis=1)

        testing_instances_in_anchor = self.get_base_model_data(rules, testing_instances_pandas_frame)
        
        index_coverage_anchors = np.where([x == self.target_class for x in testing_instances_in_anchor['label']])
        nb_instances_covered_by_rule = len(index_coverage_anchors[0])
        # Computes the number of instances from the training set that are classified as the target instance and validate the anchor rules.
        labels_instance_test_data = self.black_box_predict(self.test_data)
        index_instances_test_data_labels_as_target = np.where([x == self.target_class for x in labels_instance_test_data])
        test_coverage = nb_instances_covered_by_rule/len(index_instances_test_data_labels_as_target[0])
        instances_from_index = self.test_data[index_instances_test_data_labels_as_target]
        testing_instances_in_anchor = testing_instances_in_anchor.drop(columns=['label'])
        coverage_testing_instances_in_anchor = testing_instances_in_anchor.copy()
        nb_test_instances_in_anchor = 0
        
        for instance_index in instances_from_index:
            matches = coverage_testing_instances_in_anchor[(coverage_testing_instances_in_anchor==instance_index).all(axis=1)]
            if len(matches)>0:
                nb_test_instances_in_anchor += 1
            
        # Generates artificial instances in the area of the anchor rules until there are as many instances as in the hyperfield
        instances_in_anchor = self.generate_artificial_instances_in_anchor(testing_instances_in_anchor, nb_instances_in_sphere, instance, 
                                                rules, farthest_distance, percentage_distribution)
        labels_in_anchor = self.black_box_predict(instances_in_anchor)
        anchor_coverage = nb_test_instances_in_anchor/nb_instance_test_data_label_as_target
        if anchor_coverage != test_coverage:
          #print()
          #print()
          #print()
          #print("testing instances pandas frame befors applying", testing_instances_pandas_frame)
          #print("testing instances in anchor", testing_instances_in_anchor)
          #print("target class", self.target_class)
          #print("il y a ", nb_instances_covered_by_rule, "instances from target class validating the rules")
          #print("over", len(index_instances_test_data_labels_as_target[0]), "instances from the test set from the target class")
          #print("coverage", test_coverage)
          #print("anchor coverage", anchor_coverage)
          #print()
          #print()
          #print()
        anchor_precision = sum(labels_in_anchor == self.target_class)/len(labels_in_anchor)
        f2_anchor = (anchor_coverage+2*anchor_precision)/3
        return anchor_precision, anchor_coverage, f2_anchor, anchor_exp.names()

    def compute_lime_extending_precision_coverage(self, train_instances_in_sphere, instance_to_explain, labels_in_sphere, 
                                                growing_sphere, nb_features_employed, farthest_distance, dicrease_radius, 
                                                nb_instance_test_data_label_as_target):
        """ 
        Lime explanation and computation of precision inside the initial hypersphere
        Args: instances_in_sphere: Set of instances that are present in the hyper field
              labels_in_sphere: Corresponding labels predict by the complex model for instance in the hyper field
              growing_sphere: The growing sphere object used by APE
              nb_features_employed: Number of features used as explanation by Local Surrogate
              farthest_distance: Distance between the target instance and the farthest instances from the training data
              dicrease_radius: Ratio of dicreasing radius of the field
              nb_instance_train_data_label_as_target: Number of instances from the training data that are classify as the target instance
        Return: precision, coverage and F1 of local surrogate trained over training instances with a logistic regression model
        """
        train_instances_in_sphere, test_instances_in_sphere, labels_in_sphere, test_labels_in_sphere = train_test_split(train_instances_in_sphere, labels_in_sphere, test_size=0.4, 
                                                                    random_state=42)
        # Generate a local surrogate explanation model (centered on the closest counterfactual instance) trained over 
        # training instances with a Logistic Regression model as explanation
        ls_raw_data = self.lime_explainer.explain_instance_training_dataset(self.closest_counterfactual, self.black_box_predict, 
                                                                    num_features=nb_features_employed, model_regressor = LogisticRegression(), 
                                                                    instances_in_sphere=train_instances_in_sphere)
        prediction_inside_sphere = self.modify_instance_for_linear_model(ls_raw_data, test_instances_in_sphere)
        #print("TEST prediction inside sphere", prediction_inside_sphere)
        # Initialize the precision of Local Surrogate
        #precision_ls_raw_data = sum(test_labels_in_sphere == prediction_inside_sphere)/len(prediction_inside_sphere)
        #print("precision ls", precision_ls_raw_data)
        precision_ls_raw_data = precision_score(test_labels_in_sphere, prediction_inside_sphere)
        #print("average precision", precision_ls_raw_data)
        radius = growing_sphere.radius
        final_precision = 0

        while precision_ls_raw_data > self.threshold_precision and radius < farthest_distance:
          #print("EXTENDING the hypersphere")
            """ Extending the hypersphere radius until the precision inside the hypersphere is lower than the threshold 
            and the radius of the hyper sphere is not longer than the distances to the farthest instance from the dataset """
            final_precision = precision_ls_raw_data
            last_radius = radius
            # Extend the size of the sphere as Laugel et al. in Growing Sphere
            radius += 0.005
            position_training_instances_in_sphere, nb_training_instance_in_sphere = self.instances_from_dataset_inside_sphere(self.closest_counterfactual, 
                                                                                                                radius, self.train_data)
            train_instances_in_sphere, labels_in_sphere, percentage_distribution, _ = self.generate_instances_inside_sphere(radius, self.closest_counterfactual, self.train_data,
                                                                                                                farthest_distance, self.nb_min_instance_per_class_in_sphere,
                                                                                                                position_training_instances_in_sphere, nb_training_instance_in_sphere)
            position_testing_instances_in_sphere, nb_testing_instance_in_sphere = self.instances_from_dataset_inside_sphere(self.closest_counterfactual, 
                                                                                                                radius, self.test_data)
            test_instances_in_sphere, test_labels_in_sphere, test_percentage_distribution, _ = self.generate_instances_inside_sphere(radius, self.closest_counterfactual, self.test_data,
                                                                                                                farthest_distance, self.nb_min_instance_per_class_in_sphere,
                                                                                                                position_testing_instances_in_sphere, nb_testing_instance_in_sphere)
            # Train a new Local Surrogate explanation model on a larger hyper field (with instances inside this hyper field)
            ls_raw_data = self.lime_explainer.explain_instance_training_dataset(self.closest_counterfactual, self.black_box_predict, 
                                                                    num_features=nb_features_employed, model_regressor=LogisticRegression(), 
                                                                    instances_in_sphere=train_instances_in_sphere)
            prediction_inside_sphere = self.modify_instance_for_linear_model(ls_raw_data, test_instances_in_sphere)
            precision_ls_raw_data = self.compute_linear_regression_precision(prediction_inside_sphere, test_labels_in_sphere)
          #print("precision after extending the sphere", precision_ls_raw_data)
          #print("radius", radius)
        if final_precision > precision_ls_raw_data and precision_ls_raw_data < self.threshold_precision:
            precision_ls_raw_data = final_precision
            radius = last_radius
        
        start_time = time.time()
      #print("timing", start_time)
        position_testing_instances_in_sphere, nb_testing_instance_in_sphere = self.instances_from_dataset_inside_sphere(self.closest_counterfactual, 
                                                                                                            radius, self.test_data)
        # Compute the number of training data that are classify by the black box model as the target instance and the labels of training data instance
        nb_testing_instance_in_sphere_label_as_target, labels_testing_instance_in_sphere = self.compute_labels_inside_sphere(nb_testing_instance_in_sphere, 
                                                                                                                                position_testing_instances_in_sphere,
                                                                                                                                self.test_data)
      #print("il aura fallu %s secondes pour calculer la couverture à l'ancienne" % (time.time() - start_time))
        start_time = time.time()
        """ computation of the coverage inside the sphere for linear model on training data """
        def compute_coverage(dataset, radius):
            borne = []
            #print("min features", self.min_features)
            #print("max features", self.max_features)
            for feature, (min_feature, max_feature) in enumerate(zip(self.min_features, self.max_features)):
                range_feature = max_feature - min_feature
                #print("range feature", range_feature)
                # Modify the generation of artificial instance depending on the variance of each feature
                y = 0
                z = radius * range_feature
                #print("z", z)
                variance = self.feature_variance[feature] * radius
                #print("variance", variance)
                k = (12 * variance)**0.5
                #print("k", k)
                a1 = min(y, z - k)
                b1 = a1 + k
                #print("a1", a1)
                #print("b1", b1)
                borne.append(self.closest_counterfactual[self.continuous_features[feature]] + min(a1, b1, -a1, -b1))
                borne.append(self.closest_counterfactual[self.continuous_features[feature]] + max(a1, b1, -a1, -b1))
            #print("closest counterfactual", self.closest_counterfactual)
            #print("borne", borne)
            nb_instances_in_borne = 0
            for instance in dataset:
                borne_boolean = True
                for nb, feature_value in enumerate(instance[self.continuous_features]):
                    if borne_boolean and feature_value > borne[2*nb] and feature_value < borne[2*nb + 1]:
                        continue
                    else:
                        borne_boolean = False
                nb_instances_in_borne += 1 if borne_boolean else 0
          #print("nb instances in borne", nb_instances_in_borne, "over", len(dataset))
            return nb_instances_in_borne/len(dataset)

        lime_extending_coverage = compute_coverage(self.test_data, radius)
      #print("il aura fallu %s secondes pour calculer la couverture avec la nouvelle méthode" % (time.time() - start_time))
        lime_old_extending_coverage = nb_testing_instance_in_sphere_label_as_target/nb_instance_test_data_label_as_target
      #print("old coverage", lime_old_extending_coverage, "nb instances in field", nb_testing_instance_in_sphere_label_as_target)
        f2_lime_extending = (2*precision_ls_raw_data + lime_extending_coverage)/3
        return precision_ls_raw_data, lime_extending_coverage, f2_lime_extending, ls_raw_data.as_list(), radius

    def model_stability_index(self, instance, growing_method, opponent_class, n_instance_per_layer, first_radius, 
                            dicrease_radius, farthest_distance):
        growing_sphere = cf.CounterfactualExplanation(instance, self.black_box_predict, method=growing_method, 
                        target_class=opponent_class, continuous_features=self.continuous_features, 
                        categorical_features=self.categorical_features, 
                        categorical_values=self.categorical_values)
        growing_sphere.fit(n_in_layer=n_instance_per_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=True, 
                    verbose=self.verbose, feature_variance=self.feature_variance, 
                    farthest_distance_training_dataset=farthest_distance, 
                    probability_categorical_feature=self.probability_categorical_feature, 
                    min_counterfactual_in_sphere=self.nb_min_instance_per_class_in_sphere)
        closest_counterfactual = growing_sphere.enemy
        farthest_distance_cf = 0
        for training_instance in self.train_data:
            # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
            # but it adds a sparsity distance computation (i.e: number of same values) 
            #farthest_distance_now = get_distances(training_instance, instance, categorical_features=self.categorical_features)["euclidean"]
            farthest_distance_cf_now = distances(self.closest_counterfactual, training_instance, self)
            if farthest_distance_cf_now > farthest_distance_cf:
                farthest_distance_cf = farthest_distance_cf_now
        
        """ Generates or store instances in the area of the hyperfield and their corresponding labels """
        min_instance_per_class = self.nb_min_instance_per_class_in_sphere
        position_instances_in_sphere, nb_training_instance_in_sphere = self.instances_from_dataset_inside_sphere(closest_counterfactual, 
                                                                                                                growing_sphere.radius, self.train_data)

        instances_in_sphere, labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = self.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                                closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                                min_instance_per_class, position_instances_in_sphere, 
                                                                                                                nb_training_instance_in_sphere, libfolding=True)

        """ Compute the libfolding test to verify wheter instances in the area of the hyper sphere is multimodal or unimodal """
        if instances_in_sphere_libfolding != []:
            # In case of categorical data, we transform categorical values into probability distribution (continuous values for libfolding)
            index_counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(instances_in_sphere, self.target_class, libfolding=True)
            counterfactual_instances_in_sphere = instances_in_sphere[index_counterfactual_instances_in_sphere]
            counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
            unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), instances_in_sphere, growing_sphere.radius,
                                                         counterfactual_libfolding=counterfactual_libfolding)
        else:
            counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(instances_in_sphere, self.target_class)
            unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), instances_in_sphere, growing_sphere.radius)

        nb = 0
        while not unimodal_test:
            # While the libfolding test is not able to declare that data are multimodal or unimodal we extend the number of instances that are generated
            min_instance_per_class *= 1.5
            instances_in_sphere, labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = self.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                    closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                    min_instance_per_class, position_instances_in_sphere, 
                                                                                                    nb_training_instance_in_sphere, libfolding=True)
            
            if instances_in_sphere_libfolding != []:
                index_counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(instances_in_sphere, 
                                                                                                    self.target_class, libfolding=True)
                counterfactual_instances_in_sphere = instances_in_sphere[index_counterfactual_instances_in_sphere]
                counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
                unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), instances_in_sphere, 
                                                            growing_sphere.radius, counterfactual_libfolding=counterfactual_libfolding)
            else:
                counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(instances_in_sphere, self.target_class)
                unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), instances_in_sphere, growing_sphere.radius)
            if self.verbose:
                print("nb times libfolding is not able to determine wheter datas are unimodal or multimodal:", nb)
                print("There are ", len(counterfactual_instances_in_sphere), " instances in the datas given to libfolding.")
                print()
            nb += 1
        return self.multimodal_results

    def explain_instance(self, instance, opponent_class=None, growing_method='GF', n_instance_per_layer=2000, first_radius=0.1, 
                        nb_features_employed=6, dicrease_radius=10, all_explanations_model=False, user_experiments=False, 
                        lime_vs_local_surrogate=False, local_surrogate_experiment=False, illustrative_results=False, stability=False,
                        lime_stability=False, k_closest=False, model_stability_index=False, nb_iteration=0):
        """
        Returns either an explanation from anchors or lime along with one or multiple counter factual explanation
        Args: instance: Target instance to explain
              opponent_class: Class of the desired counterfactual instance
              growing_method: Type of method to find counterfactual instances (GF = GrowingFields; GS = GrowingSpheres)
              n_instance_per_layer: Number of instances require in each layer for growing field
              first_radius: Radius of the initial field for growing field
              nb_features_employed: Indicate how many features will be used as explanation for the linear explanation (used also for experiments)
              dicrease_radius: Ratio of dicreasing the radius of the growing field
              all_explanations_model: generate explanation with multiple explanation models (for experiments)
              user_experiments: return features employed by linear and rule based explanation (for experiments)
              lime_vs_local_surrogate: Return features employed by Lime and LS (for experiments)
              local_surrogate_experiment: Compute multiple local surrogate explanations and return precision, coverage and F1 (for experiments)
        Return: APE's coverage
                APE's precision
                APE's F1
                Indicate whether counter factual instances are multimodal: 1 or unimodal: 0
        """
        self.target_class = self.black_box_predict(instance.reshape(1, -1))[0]
        # Computes the distance to the farthest instance from the training dataset to bound generating instances 
        farthest_distance = 0
        for training_instance in self.train_data:
            # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
            # but it adds a sparsity distance computation (i.e: number of same values) 
            #farthest_distance_now = get_distances(training_instance, instance, categorical_features=self.categorical_features)["euclidean"]
            farthest_distance_now = distances(training_instance, instance, self)
            if farthest_distance_now > farthest_distance:
                farthest_distance = farthest_distance_now
        start_time = time.time()
      #print("farthest distance", farthest_distance)
        growing_sphere = cf.CounterfactualExplanation(instance, self.black_box_predict, method=growing_method, target_class=opponent_class, 
                                                    continuous_features=self.continuous_features, 
                                                    categorical_features=self.categorical_features, 
                                                    categorical_values=self.categorical_values,
                                                    max_features=self.max_features,
                                                    min_features=self.min_features)
        growing_sphere.fit(n_in_layer=n_instance_per_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=True, 
                                                    verbose=self.verbose, feature_variance=self.feature_variance, 
                                                    farthest_distance_training_dataset=farthest_distance, 
                                                    probability_categorical_feature=self.probability_categorical_feature, 
                                                    min_counterfactual_in_sphere=self.nb_min_instance_per_class_in_sphere)
      #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
      #print("done searching for closest counterfactual")
        self.closest_counterfactual = growing_sphere.enemy
        # Computes the distance to the farthest instance from the training dataset to bound generating instances 
        farthest_distance_cf = 0
        for training_instance in self.train_data:
            # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
            # but it adds a sparsity distance computation (i.e: number of same values) 
            #farthest_distance_now = get_distances(training_instance, instance, categorical_features=self.categorical_features)["euclidean"]
            farthest_distance_cf_now = distances(self.closest_counterfactual, training_instance, self)
            if farthest_distance_cf_now > farthest_distance_cf:
                farthest_distance_cf = farthest_distance_cf_now
      #print("farthest distance from counterfactual", farthest_distance_cf)
        if self.verbose:
            print("The farthest instance from the training dataset is ", farthest_distance, " away from the target.")
            if opponent_class == None:
                opponent_class = self.black_box_predict(growing_sphere.enemy.reshape(1, -1))[0]
                print("Class of the closest counterfactual: ", opponent_class)
            print("radius of the hyperfield:", growing_sphere.radius)
            print("The target instance to explain is ", instance)
            print("The associated closest counterfactual is ", self.closest_counterfactual)
        
        """ Generates or store instances in the area of the hyperfield and their corresponding labels """
        min_instance_per_class = self.nb_min_instance_per_class_in_sphere
      #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        position_training_instances_in_sphere, nb_training_instance_in_sphere = self.instances_from_dataset_inside_sphere(self.closest_counterfactual, 
                                                                                                    growing_sphere.radius, self.train_data)
      #print("DONE INSTANCES FROM DATASET INSIDE SPHERE--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        training_instances_in_sphere, train_labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = self.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                    self.closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                    min_instance_per_class, position_training_instances_in_sphere, 
                                                                                                    nb_training_instance_in_sphere, libfolding=True)
      #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
      #print("done generating train instances")
        position_testing_instances_in_sphere, nb_testing_instance_in_sphere = self.instances_from_dataset_inside_sphere(self.closest_counterfactual, 
                                                                                                                growing_sphere.radius, self.test_data)
        test_instances_in_sphere, test_labels_in_sphere, test_percentage_distribution, _ = self.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                    self.closest_counterfactual, 
                                                                                                    self.test_data,
                                                                                                    farthest_distance_cf, 
                                                                                                    self.nb_min_instance_per_class_in_sphere,
                                                                                                    position_testing_instances_in_sphere, 
                                                                                                    nb_testing_instance_in_sphere)
      #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
      #print("done generating test instances")
        if local_surrogate_experiment:
            local_surrogate_precision, local_surrogate_coverage, f2_local_surrogate = compute_local_surrogate_precision_coverage(self, 
                                                instance, growing_sphere,
                                                test_instances_in_sphere, test_labels_in_sphere,
                                                position_testing_instances_in_sphere, nb_testing_instance_in_sphere, 
                                                nb_features_employed)
            return local_surrogate_precision, local_surrogate_coverage, f2_local_surrogate

        """ Compute the libfolding test to verify wheter instances in the area of the hyper sphere is multimodal or unimodal """
        if instances_in_sphere_libfolding != []:
            # In case of categorical data, we transform categorical values into probability distribution (continuous values for libfolding)
            index_counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(training_instances_in_sphere,
                                                                                         self.target_class, libfolding=True)
          #print("done storing counterfactual in hyper field")
          #print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            counterfactual_instances_in_sphere = training_instances_in_sphere[index_counterfactual_instances_in_sphere]
          #print("done finding instances in hyper field")
          #print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
          #print("start of unimodality test")
            unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius,
                                                         counterfactual_libfolding=counterfactual_libfolding)
        else:
            counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(training_instances_in_sphere, self.target_class)
            unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius)

      #print("test unimodal done")
      #print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        nb = 0
        while not unimodal_test:
            # While the libfolding test is not able to declare that data are multimodal or unimodal we extend the number of instances that are generated
            min_instance_per_class *= 1.5
            training_instances_in_sphere, train_labels_in_sphere, percentage_distribution, instances_in_sphere_libfolding = self.generate_instances_inside_sphere(growing_sphere.radius, 
                                                                                                    self.closest_counterfactual, self.train_data, farthest_distance_cf, 
                                                                                                    min_instance_per_class, position_training_instances_in_sphere, 
                                                                                                    nb_training_instance_in_sphere, libfolding=True)
            
            if instances_in_sphere_libfolding != []:
                index_counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(training_instances_in_sphere, 
                                                                                                self.target_class, libfolding=True)
                counterfactual_instances_in_sphere = training_instances_in_sphere[index_counterfactual_instances_in_sphere]
                counterfactual_libfolding = instances_in_sphere_libfolding[index_counterfactual_instances_in_sphere]
                unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), 
                                                                training_instances_in_sphere, growing_sphere.radius,
                                                                counterfactual_libfolding=counterfactual_libfolding)
            else:
                counterfactual_instances_in_sphere = self.store_counterfactual_instances_in_sphere(training_instances_in_sphere, self.target_class)
                unimodal_test = self.check_test_unimodal_data(np.array(counterfactual_instances_in_sphere), training_instances_in_sphere, growing_sphere.radius)
            if self.verbose:
                print("nb times libfolding is not able to determine wheter datas are unimodal or multimodal:", nb)
                print("There are ", len(counterfactual_instances_in_sphere), " instances in the datas given to libfolding.")
                print()
            nb += 1
        
        if model_stability_index or lime_stability:
            print("compute model stability index...")
            model_stability = []
            initial_multimodal = self.multimodal_results
            model_stability.append(initial_multimodal)
            for i in range(10):
                model_stability.append(self.model_stability_index(instance, growing_method, 
                                        opponent_class, n_instance_per_layer, first_radius, 
                                        dicrease_radius, farthest_distance))
            model_stability_score = model_stability.count(initial_multimodal)

        """ Computes the labels for instances from the dataset to compute precision for explanation method """
        labels_instance_test_data = self.black_box_predict(self.test_data)
        nb_instance_test_data_label_as_target = sum(x == self.target_class for x in labels_instance_test_data)
        
        """ Different cases for experiments """
        if user_experiments:
            return simulate_user_experiments(self, instance, nb_features_employed, farthest_distance_cf, 
                                            self.closest_counterfactual, growing_sphere,
                                            position_training_instances_in_sphere, nb_training_instance_in_sphere)

        elif lime_vs_local_surrogate:
            return simulate_user_experiments_lime_ls(self, instance, nb_features_employed, farthest_distance_cf, 
                                            self.closest_counterfactual, growing_sphere,
                                            position_training_instances_in_sphere, nb_training_instance_in_sphere)

        elif all_explanations_model:
            return compute_all_explanation_method_precision(self, instance, growing_sphere, dicrease_radius, growing_sphere.radius,
                                            nb_training_instance_in_sphere, nb_instance_test_data_label_as_target,
                                            position_training_instances_in_sphere, training_instances_in_sphere, train_labels_in_sphere,
                                            farthest_distance, farthest_distance_cf, percentage_distribution, nb_features_employed) 
        elif illustrative_results:
            return ape_illustrative_results(self, instance, counterfactual_instances_in_sphere)

        elif stability:
            features_employed_in_linear, features_employed_by_ape, features_employed_in_rule = simulate_user_experiments(self, 
                                            instance, nb_features_employed, farthest_distance_cf, self.closest_counterfactual, 
                                            growing_sphere, position_training_instances_in_sphere, nb_training_instance_in_sphere)
            multimodal = 1 if self.multimodal_results else 0
            return multimodal, features_employed_by_ape
        
        elif lime_stability:
            csi_ls, vsi_ls = self.lime_explainer.check_stability(self.closest_counterfactual, self.black_box_predict_proba, n_calls=10, index_verbose=False)
            
            def compute_vsi_anchors(used_features):
                feature_ids = used_features
                used_features = [self.feature_names[i] for i in feature_ids]
                conf_int = {}
                for test in used_features:
                    conf_int[test] = 1
                return conf_int

            confidence_intervals = []
            for i in range(10):
                _, _, features_employed_in_rule = simulate_user_experiments(self, 
                                            instance, nb_features_employed, farthest_distance_cf, self.closest_counterfactual, 
                                            growing_sphere, position_training_instances_in_sphere, nb_training_instance_in_sphere)
                #print("features employed in rule", features_employed_in_rule)
                confidence_intervals.append(compute_vsi_anchors(features_employed_in_rule))
            #print("confidence intervals anchors", confidence_intervals)
            csi_anchors, vsi_anchors = compare_confints(confidence_intervals=confidence_intervals,
                                    index_verbose=True, anchors=True)

            print("csi / vsi anchor", csi_anchors, vsi_anchors)
            if self.multimodal_results:
                vsi = [vsi_anchors, vsi_ls, vsi_anchors]
                csi = [None, csi_ls, None]
            else:
                vsi = [vsi_ls, vsi_ls, vsi_anchors]
                csi = [csi_ls, csi_ls, None]
            return model_stability_score, csi, vsi

        elif k_closest is not False:
            #closest_instances = np.concatenate((instances_in_sphere, self.train_data), axis=0)
            if 'S' in growing_method:
                growing_method_other = 'GF'
            else:
                growing_method_other = 'GS'
            
            index_train_data_counterfactual_class = np.where([x != self.target_class for x in self.black_box_predict(self.train_data)])
            train_data_counterfactual_class = self.train_data[index_train_data_counterfactual_class]

            neigh = NearestNeighbors(n_neighbors=k_closest+1, algorithm='ball_tree', metric=distances, metric_params={"ape": self})
            if nb_iteration == 0:
                neigh.fit(train_data_counterfactual_class, self.black_box_predict(train_data_counterfactual_class))
                #kdt = scipy.spatial.cKDTree(train_data_counterfactual_class)
                try:
                    dists, neighs = neigh.kneighbors(self.clusters_centers, k_closest+1)
                  #print("dists, neighs of knn", dists, neighs)
                    #dists, neighs = kdt.query(self.clusters_centers, k_closest+1)
                    #print("dists, neighs of tree", dists, neighs)
                except AttributeError:
                    dists, neighs = neigh.kneighbors(self.closest_counterfactual.reshape(1, -1), k_closest+1)
                    #dists, neighs = kdt.query(self.closest_counterfactual.reshape(1, -1), k_closest+1)
                mean_dists, mean_neighs = neigh.kneighbors(train_data_counterfactual_class, k_closest+1)
                #mean_dists, mean_neighs = kdt.query(train_data_counterfactual_class, k_closest+1)

                closest_ennemis = train_data_counterfactual_class[neighs[0]]
                dists = []
                for ennemis in closest_ennemis:
                    dists.append(distances(self.closest_counterfactual, ennemis, self))
                
                avg_dists = np.mean(dists)
                mean_avg_dists = np.mean(mean_dists[:, 1:], axis=1)
                avg_dists_other, avg_dists_all_other = self.explain_instance(instance, growing_method=growing_method_other, 
                                                            k_closest=k_closest, nb_iteration=nb_iteration+1)
                return avg_dists, np.mean(mean_avg_dists), avg_dists_other, avg_dists_all_other
            else:
                #kdt = scipy.spatial.cKDTree(train_data_counterfactual_class)
                try:
                    #dists, neighs = kdt.query(self.clusters_centers, k_closest+1)
                    dists, neighs = neigh.kneighbors(self.clusters_centers, k_closest+1)
                except AttributeError:
                    #dists, neighs = kdt.query(self.closest_counterfactual.reshape(1, -1), k_closest+1)
                    dists, neighs = neigh.kneighbors(self.closest_counterfactual.reshape(1, -1), k_closest+1)
                
                closest_ennemis = train_data_counterfactual_class[neighs[0]]
                dists = []
                for ennemis in closest_ennemis:
                    dists.append(distances(self.closest_counterfactual, ennemis, self))

                #mean_dists, mean_neighs = kdt.query(train_data_counterfactual_class, k_closest+1)
                mean_dists, mean_neighs = neigh.kneighbors(train_data_counterfactual_class, k_closest+1)
                avg_dists = np.mean(dists)
                mean_avg_dists = np.mean(mean_dists[:, 1:], axis=1)
                return avg_dists, np.mean(mean_avg_dists)


        if self.multimodal_results:
            # In case of multimodal data, we generate a rule based explanation and compute precision and coverage of this explanation model
            ape_precision, ape_coverage, ape_2, ape_explanation = self.compute_anchor_precision_coverage(instance, 
                                        labels_instance_test_data, len(test_instances_in_sphere), 
                                        farthest_distance, percentage_distribution, nb_instance_test_data_label_as_target)
        
        else:
            # In case of unimodal data, we generate linear explanation and compute precision and coverage of this explanation model
            ape_precision, ape_coverage, ape_f2, ape_explanation, self.extended_radius = self.compute_lime_extending_precision_coverage(training_instances_in_sphere, 
                                        instance, train_labels_in_sphere, growing_sphere, nb_features_employed,
                                        farthest_distance_cf, dicrease_radius, nb_instance_test_data_label_as_target)

        return ape_coverage, ape_precision, ape_f2, 1 if self.multimodal_results else 0
