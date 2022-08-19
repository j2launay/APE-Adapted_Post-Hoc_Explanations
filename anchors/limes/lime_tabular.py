"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
import json
import warnings

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state

from .discretize import QuartileDiscretizer
from .discretize import DecileDiscretizer
from .discretize import EntropyDiscretizer
from .discretize import MDLPDiscretizer
from .discretize import KMeansDiscretizer
from .discretize import BaseDiscretizer
from . import explanation
from . import lime_base
from .utils_stability import refactor_confints_todict, compute_WLS_stdevs, compare_confints, LocalModelError
import sys
sys.path.append("...")
from growingspheres.utils.gs_utils import distances as dista

class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        out_list = list(zip(self.exp_feature_names,
                            self.feature_values,
                            weights))
        if not show_all:
            out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 random_state=None,
                 filename=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'MDLP', 'quartile', 'decile', 'entropy', 'kmeans' or a 
                BaseDiscretizer instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)
        self.training_data = training_data
        self.training_labels = training_labels

        self.discretizer = None
        if discretize_continuous:
            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'MDLP':
                self.discretizer = MDLPDiscretizer(
                        training_data, self.categorical_features, 
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state) 
            elif discretizer == 'kmeans':
                self.disc = KMeansDiscretizer(train_data, 
                        self.categorical_features, 
                        self.feature_names, filename)  
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))
            discretized_training_data = self.discretizer.discretize(
                    training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)
        self.scaler = None
        self.class_names = class_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            feature_count = collections.defaultdict(lambda: 0.0)
            column = training_data[:, feature]
            if self.discretizer is not None:
                column = discretized_training_data[:, feature]
                feature_count[0] = 0.
                feature_count[1] = 0.
                feature_count[2] = 0.
                feature_count[3] = 0.
            for value in column:
                feature_count[value] += 1
            values, frequencies = map(list, zip(*(feature_count.items())))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 sum(frequencies))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         stability=False):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        data, inverse = self.__data_inverse(data_row, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()
        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                """
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
                """

            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred, 
             #ret_exp.data_print, ret_exp.prediction_print) 
             ret_exp.easy_model, ret_exp.used_features)= self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection,
                    stability=stability)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def explain_instance_training_dataset(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         instances_in_sphere=[],
                         ape=None,
                         user_simulated=False):
        """Generates explanations for a prediction.

        First, we learn locally weighted linear models on 
        the instances from "instances_in_sphere" to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            instances_in_sphere: Matrix of instances on which we train the 
                linear model. If None we use the training dataset

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        #data, inverse = self.__data_inverse(data_row, num_samples)
        if instances_in_sphere != []:
            data, inverse = instances_in_sphere, instances_in_sphere
        else:
            data, inverse = self.training_data, self.training_data
        
        np.insert(data, 0, data_row)
        np.insert(inverse, 0, data_row)

        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        if ape is None:
            distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()
        else:
            distances = []
            for instance in instances_in_sphere:
                distances.append(dista(scaled_data[0], instance, ape))
            distances = np.array(distances)
            

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                """
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
                """

            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                try:
                    name = self.categorical_names[i][name]
                except IndexError:
                    print("error in dataset categorical names", self.categorical_names)
                    print(i)
                    print(name)
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]
        """
        print("domain mapper")
        print("feature names", feature_names)
        print("ape encoded features names", ape.encoded_features_names)
        print("values", values)
        print("scaled data", scaled_data[0])
        print("categorical features", categorical_features)
        print("discretized", discretized_feature_names)
        """
        if ape.categorical_features == []:
            domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names)
        
        else:
            domain_feature_names = ape.encoded_features_names
            #print(ape.enc.transform(scaled_data[0][ape.categorical_features].reshape(1, -1)).toarray())
            #print(list(range(len(ape.continuous_features))))
            domain_values = np.append(ape.enc.transform(scaled_data[0][ape.categorical_features].reshape(1, -1)).toarray()[0], 
                                                np.ones(len(ape.continuous_features)), axis=0)
            #domain_categorical_features = ape.enc.transform(scaled_data[0][ape.categorical_features].reshape(1, -1)).toarray()[0]
            """
            print("domain_categorical_features", domain_categorical_features)
            print("domain values", domain_values)
            print("target instance", scaled_data[0][ape.categorical_features].reshape(1, -1))
            print("TEST", range(len(scaled_data[0]) - len(ape.continuous_features)))
            print("2ND", range(len(domain_values) - len(ape.continuous_features)))
            #print("2DN", discretized_feature_names[:,ape.continuous_features])"""
            domain_mapper = TableDomainMapper(domain_feature_names,
                                            domain_values,
                                            ape.enc.transform(scaled_data[0][ape.categorical_features].reshape(1, -1)).toarray(),
                                            categorical_features=range(len(scaled_data[0][ape.categorical_features].reshape(1, -1)) - len(ape.continuous_features)),
                                            discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred, 
             #ret_exp.data_print, ret_exp.prediction_print) 
             ret_exp.easy_model, ret_exp.used_features)= self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection,
                    ape=ape)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
        if user_simulated:
            return ret_exp, self.base.used_features
        return ret_exp

    def data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        data = np.zeros((num_samples, data_row.shape[0]))
        categorical_features = range(data_row.shape[0])
        if self.discretizer is None:
            data = self.random_state.normal(
                    0, 1, num_samples * data_row.shape[0]).reshape(
                    num_samples, data_row.shape[0])
            data = data * self.scaler.scale_ + self.scaler.mean_
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        data = np.zeros((num_samples, data_row.shape[0]))
        categorical_features = range(data_row.shape[0])
        if self.discretizer is None:
            data = self.random_state.normal(
                    0, 1, num_samples * data_row.shape[0]).reshape(
                    num_samples, data_row.shape[0])
            data = data * self.scaler.scale_ + self.scaler.mean_
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse
    
    def confidence_intervals(self):
        """
        Method to calculate stability indices of an application of the LIME method
        to a particular unit of the dataset.
        The stability indices are described in the paper:
        "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models"
        which can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
        The paper is currently under review in the Journal of Operational Research Society (JORS)
        Returns:
            conf_int: list of two values (lower and upper bound of the confidence interval)
        """

        stdevs_beta = compute_WLS_stdevs(X=self.base.X, Y=self.base.true_labels,
                                         weights=self.base.weights, alpha=self.base.alpha)

        beta_ridge = self.base.easy_model.coef_.tolist()
        feature_ids = self.base.used_features
        used_features = [self.feature_names[i] for i in feature_ids]

        conf_int = refactor_confints_todict(means=beta_ridge, st_devs=stdevs_beta, feat_names=used_features)

        return conf_int

    def check_stability(self,
                        data_row,
                        predict_fn,
                        labels=(1,),
                        top_labels=None,
                        num_features=10,
                        num_samples=5000,
                        distance_metric='euclidean',
                        model_regressor=None,
                        n_calls=10,
                        index_verbose=False,
                        verbose=False,
                        ls=False,
                        ape=None,
                        instances_in_sphere=[]):

        """
        Method to calculate stability indices for a trained LIME instance.
        The stability indices are relative to the particular data point we are explaining.
        The stability indices are described in the paper:
        "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models".
        It can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
        The paper is currently under review in the Journal of Operational Research Society (JORS)
        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase.
                If not Ridge Regression, the stability indices may not be calculated
                and the method will raise a LocalModelError.
            n_calls: number of repeated Lime calls to perform.
                High number of calls slows down the execution,
                however a meaningful comparison takes place only when
                there is a reasonable number of calls to compare.
            index_verbose: Controls for the verbosity at the stability indices level,
                when set to True gives information about partial values related to stability.
            verbose: Controls for the verbosity at the LocalModel level,
                when set to True, gives information about the repeated calls of WRR.
        Returns:
            csi: index to evaluate the stability of the coefficients of each variable across
            different Lime explanations obtained from the repeated n_calls.
            Ranges from 0 to 100.
            vsi: index to evaluate whether the variables retrieved in different Lime explanations
                are the same. Ranges from 0 to 100.
        """

        # Override verbosity in the LimeBaseOvr instance
        previous_verbose = self.base.verbose
        self.base.verbose = verbose

        confidence_intervals = []
        print("searching for vsi and csi indicators...")
        for i in range(n_calls):
            if ls :
                alpha = self.explain_instance_training_dataset(data_row,
                                  predict_fn,
                                  labels,
                                  top_labels,
                                  num_features,
                                  num_samples,
                                  distance_metric,
                                  model_regressor,
                                  instances_in_sphere=instances_in_sphere,
                                  ape=ape,
                                  stability=True).easy_model.alpha
            else:
                alpha = self.explain_instance(data_row,
                                  predict_fn,
                                  labels,
                                  top_labels,
                                  num_features,
                                  num_samples,
                                  distance_metric,
                                  model_regressor,
                                  stability=True).easy_model.alpha

            # The first time check if the local model is WRR, otherwise raise Exception
            if not i:
                if alpha is None:
                    raise LocalModelError("""Lime Local Model is not a Weighted Ridge Regression (WRR),
                    Stability indices may not be computed: the formula is model specific""")
            
            confidence_intervals.append(self.confidence_intervals())

        csi, vsi = compare_confints(confidence_intervals=confidence_intervals,
                                    index_verbose=index_verbose)

        # Set back the original verbosity
        self.base.verbose = previous_verbose

        return csi, vsi