# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import check_random_state

from . import growingfields, growingspheres


class CounterfactualExplanation:
    """
    Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
    """
    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None,
                continuous_features=None, categorical_features=[], categorical_values=[], max_features=[], min_features=[]):
        """
        Init function
        obs_to_interprete: the instance for which it searches the closest counterfactual
        prediction_fn: black-box classification function
        method: algorithm to use ('GS' for GrowingSpheres and 'GF' for GrowingFields)
        If target_class is None it returns the counterfactual from the closest class, otherwise it returns the counterfactual of the target class given
        Continuous features, categorical features and categorical values are the list of features or values that will be used to transform growing sphere into growing field
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        if method == 'GS':
                self.methods_ = {'GS': growingspheres.GrowingSpheres}
        else:
                self.methods_ = {'GF': growingfields.GrowingFields}
        self.fitted = 0
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.max_features = max_features
        self.min_features = min_features
        
    def fit(self, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False, 
                        feature_variance=None, farthest_distance_training_dataset=None, probability_categorical_feature=None,
                        min_counterfactual_in_sphere=0):
        """
        find the counterfactual with the specified method
        """
        cf = self.methods_[self.method](self.obs_to_interprete,
                self.prediction_fn,
                feature_variance=feature_variance,
                max_features=self.max_features,
                min_features=self.min_features,
                target_class=self.target_class,
                n_in_layer=n_in_layer,
                first_radius=first_radius,
                dicrease_radius=dicrease_radius,
                sparse=sparse,
                verbose=verbose,
                continuous_features=self.continuous_features, 
                categorical_features=self.categorical_features, 
                categorical_values=self.categorical_values,
                farthest_distance_training_dataset=farthest_distance_training_dataset,
                probability_categorical_feature=probability_categorical_feature,
                min_counterfactual_in_sphere=min_counterfactual_in_sphere)
        self.enemy, self.onevsrest, self.radius = cf.find_counterfactual()
        self.e_star = cf.e_star
        self.move = self.enemy - self.obs_to_interprete
        self.fitted = 1
