"""
Contains abstract functionality for learning locally linear sparse model.
"""
from __future__ import print_function
import numpy as np
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from collections import Counter
import random

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features, model_regressor=None):
        """Iteratively adds features to the model"""
        if model_regressor is None:
            model_regressor = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        clf = model_regressor
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels, 
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method, 
                model_regressor=None):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if model_regressor is None:
            model_regressor = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features, model_regressor=model_regressor)
        elif method == 'highest_weights':
            clf = model_regressor
            clf.fit(data, labels, sample_weight=weights)
            if clf.coef_.ndim > 1:
                clf.coef_ = clf.coef_[0]
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method, model_regressor=model_regressor)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   stability=False,
                                   ape=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
        """
        weights = self.kernel_fn(distances)
        #logistic = False
        if model_regressor is not None and neighborhood_labels.ndim == 1:
            #logistic = True
            labels_column = neighborhood_labels
        elif model_regressor is not None:   
            labels_column = []
            for neighborhood_label in neighborhood_labels[:,label]:
                if neighborhood_label > 0.5:
                    labels_column.append(label)
                else:
                    labels_column.append(1-label)
        else:
            labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection, 
                                               model_regressor=model_regressor)

        self.used_features = used_features
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        if ape.categorical_features != []:
            #print("neighborod data", neighborhood_data[0])
            #print("categorical features from ape in lime base", ape.categorical_features)
            #c = Counter(neighborhood_labels)
            neighborhood_data_index_minorities_class = np.where([x == label for x in neighborhood_labels])[0]
            if len(neighborhood_labels) > 2 * len(neighborhood_data_index_minorities_class):
                test = neighborhood_data[neighborhood_data_index_minorities_class]
                how_many = len(neighborhood_labels) - len(neighborhood_data_index_minorities_class)
                idx = np.random.randint(len(neighborhood_data_index_minorities_class), size=how_many)
                add_index_for_oversampling = [label]*how_many
                weights_values_for_oversampling = weights[neighborhood_data_index_minorities_class]
            else:
                index_class_counterfactual = list(set(list(range(0, len(neighborhood_labels)))) - set(neighborhood_data_index_minorities_class))
                test = neighborhood_data[index_class_counterfactual]
                how_many =  len(neighborhood_data_index_minorities_class) - (len(neighborhood_labels) - len(neighborhood_data_index_minorities_class))
                idx = np.random.randint(len(index_class_counterfactual), size=how_many)
                add_index_for_oversampling = [1-label]*how_many
                weights_values_for_oversampling = weights[index_class_counterfactual]
            add_instance_for_oversampling = test[idx,:]
            add_sample_weight = weights_values_for_oversampling[idx]
            weights = np.concatenate((weights, add_sample_weight))
            neighborhood_data = np.concatenate((neighborhood_data, add_instance_for_oversampling))
            labels_column = np.concatenate((neighborhood_labels, add_index_for_oversampling))
            #print("neighborhood labels after", neighborhood_labels)
            codes = ape.enc.transform(neighborhood_data[:,ape.categorical_features]).toarray()
            """train_enc = train_data[:,categorical_features]
            self.enc.fit(train_enc)
            #self.enc.fit(train_data)
            codes = self.enc.transform(train_enc).toarray()
            #codes = self.enc.transform(train_data).toarray()"""
            #categorical_features_names = []
            #for i in categorical_features:
            #    categorical_features_names.append(feature_names[i])
            neighborhood_data = np.append(np.asarray(codes), 
                                neighborhood_data[:,ape.continuous_features], axis=1)
            #print("neighborhood data after", neighborhood_data[0])
            used_features = []
            for i in range(len(neighborhood_data[0])):
                used_features.append(i)
            #print("used feature after", used_features)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                        labels_column, sample_weight=weights)
        
        prediction_score = easy_model.score(
                        neighborhood_data[:, used_features],
                        labels_column, sample_weight=weights)
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if easy_model.coef_.ndim > 1:
            coef = easy_model.coef_[0]
        else:
            coef = easy_model.coef_
        """if len(easy_model.coef_) > 1 and logistic:
            # Case for multi class
            coef = easy_model.coef_[local_pred]
        else:
            coef = easy_model.coef_"""
        #print("coef for easy model", coef)
        #print("label to use", used_features)        
        #inverse_features = ape.enc.inverse_transform(np.asarray(used_features).reshape(1, -1))
        #print("TEST", inverse_features)
        if stability:
            # For Lime stability computation
            print("searching for vsi and csi indicators...")
            try:
                assert isinstance(easy_model, Ridge)
            except AssertionError:
                self.alpha = None
                print("""Attention: Lime Local Model is not a Weighted Ridge Regression (WRR),
                Lime Method will work anyway, but the stability indices may not be computed
                (the formula is model specific)""")
            else:
                self.alpha = easy_model.alpha
            finally:
                self.easy_model = easy_model
                self.X = neighborhood_data[:, used_features]
                self.weights = weights
                self.true_labels = labels_column
        return (easy_model.intercept_,
                sorted(zip(used_features, coef),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred, easy_model, used_features)