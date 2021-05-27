# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import check_random_state

from .utils.gs_utils import get_distances
from . import growingspheres


class CounterfactualExplanation:
    """
    Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
    """
    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None,
                continuous_features=None, categorical_features=[], categorical_values=[]):
        """
        Init function
        method: algorithm to use
        random_state
        If target_class is None it returns the counterfactual from the closest class, otherwise it returns the counterfactual of the target class given
        Continuous features, categorical features and categorical values are the list of features or values that will be used to transform growing sphere into growing field
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        self.methods_ = {'GS': growingspheres.GrowingSpheres}
        self.fitted = 0
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        
    def fit(self, caps=None, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False, 
                        feature_variance=None, farthest_distance_training_dataset=None, probability_categorical_feature=None,
                        min_counterfactual_in_sphere=0):
        """
        find the counterfactual with the specified method
        """
        cf = self.methods_[self.method](self.obs_to_interprete,
                self.prediction_fn,
                self.target_class,
                caps,
                n_in_layer,
                first_radius,
                dicrease_radius,
                sparse,
                verbose,
                continuous_features=self.continuous_features, 
                categorical_features=self.categorical_features, 
                categorical_values=self.categorical_values,
                feature_variance=feature_variance,
                farthest_distance_training_dataset=farthest_distance_training_dataset,
                probability_categorical_feature=probability_categorical_feature,
                min_counterfactual_in_sphere=min_counterfactual_in_sphere)
        self.enemy, self.onevsrest, self.radius = cf.find_counterfactual()
        self.e_star = cf.e_star
        self.move = self.enemy - self.obs_to_interprete
        self.fitted = 1

    def distances(self, metrics=None):
        """
        scores de distances entre l'obs et le counterfactual
        """
        if self.fitted < 1:
            raise AttributeError('CounterfactualExplanation has to be fitted first!')
        return get_distances(self.obs_to_interprete, self.enemy, metrics=metrics)
    