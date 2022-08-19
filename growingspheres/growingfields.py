#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils.gs_utils import generate_inside_field, generate_categoric_inside_ball, distances, get_distances
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state


class GrowingFields:
    """
    class to fit the Growing Fields algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    max_features: array of maximum value for each feature from the training set
    min_features: array of minimum value for each feature from the training set
    feature_variance: array of the variance for each feature from the training set
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                max_features,
                min_features,
                feature_variance,
                target_class=None,
                caps=None,
                n_in_layer=2000,
                first_radius=0.01,
                dicrease_radius=2,
                sparse=True,
                verbose=False,
                continuous_features=None,
                categorical_features=[],
                categorical_values=[],
                farthest_distance_training_dataset=None,
                probability_categorical_feature=None,
                min_counterfactual_in_sphere=0
                ):
        """
        Args: obs_to_interprete: Raw instance for which we generate a counterfactual explanation
              predicition_fn: Function used by the black box model to classify instances
              target_class: If target_class is None it returns the counterfactual from the closest class, 
                            otherwise it returns the counterfactual of the given target class
              n_in_layer: number of instances generated in each sphere (or field)
              first_radius: Initial radius of the sphere (or field)
              dicrease_radius: ratio to dicrease the radius of the sphere (or field)
              sparse: If set to True execute a feature selection
              verbose: If set to True print maximum information about the research of the closest counterfactual
              continuous_features: List of features that are continuous or discrete
              categorical_features: List of features that are categorical
              categorical values: Array of arrays containing the values for each categorical feature
              feature_variance: Array of variance for each continuous feature
              farthest_distance_training_dataset: Distance from the instance to explain to the farthest distance from the training data
              probability_categorical_feature: Maximum percentage of the distribution to change the value of artificial instance for categorical feature
              min_counterfactual_in_sphere: Minimum number of counterfactual instances to find in the sphere (or field) to stop the algorithm
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        if target_class == None:
            self.target_other = True
            target_class = self.y_obs
        else:
            self.target_other = False
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius
        self.sparse = sparse
        self.verbose = verbose
        self.continuous_features = continuous_features if continuous_features != None else [range(len(obs_to_interprete))]
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.feature_variance = feature_variance
        self.farthest_distance_training_dataset = farthest_distance_training_dataset
        self.probability_categorical_feature = probability_categorical_feature
        self.min_counterfactual_in_sphere = min_counterfactual_in_sphere
        self.max_features = max_features
        self.min_features = min_features
        
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

        
    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        ennemies_, radius = self.exploration()
        ennemies_ = sorted(ennemies_, 
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))
        closest_ennemy_ = ennemies_[0]
        self.e_star = closest_ennemy_
        if self.sparse == True:
            out = self.feature_selection(closest_ennemy_)
        else:
            out = closest_ennemy_
        return out, ennemies_, radius
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_ = 999
        radius_ = self.first_radius
        
        while n_ennemies_ > 0:
            first_layer_ = self.ennemies_in_layer_((0, radius_), self.caps, self.n_in_layer)
            n_ennemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.dicrease_radius
            if self.verbose == True:
                print("%d ennemies found in initial sphere. Zooming in..."%n_ennemies_)
            
        else:
            if self.verbose == True:
                print("Exploring...") 
            step_ = (self.dicrease_radius - 1) * radius_/2.0
            while n_ennemies_ <= self.min_counterfactual_in_sphere:
                step_ = min(radius_ / 10, step_* 2)
                layer = self.ennemies_in_layer_((radius_, radius_ + step_), self.caps, self.n_in_layer)
                n_ennemies_ = layer.shape[0]
                radius_ = min(1, radius_ + step_)
                if (radius_ == 1) and n_ennemies_ == 0:
                    return True
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return layer, radius_
    
    
    def ennemies_in_layer_(self, segment, caps=None, n=1000):
        """
        Basis for GF: generates a hyperfield layer, labels it with the blackbox 
        and returns the instances that are predicted to belong to the target class.
        """
        if self.categorical_features != []:
            # If there are categorical features we must have a maximum distribution probability for changing the values of 
            # categorical feature for artificial instances
            if self.farthest_distance_training_dataset is None:
                print("you must initialize a distance for the percentage distribution")
            else:
                # Initialize the percentage of categorical features that are changed 
                percentage_distribution = segment[1]/self.farthest_distance_training_dataset*100
            layer = generate_categoric_inside_ball(center= self.obs_to_interprete, segment=segment, n=n, 
                                            percentage_distribution=percentage_distribution,continuous_features=self.continuous_features, 
                                            categorical_features=self.categorical_features, categorical_values=self.categorical_values, 
                                            feature_variance= self.feature_variance, probability_categorical_feature=self.probability_categorical_feature,
                                            min_features=self.min_features, max_features=self.max_features)
        else:
            layer = generate_inside_field(self.obs_to_interprete, segment, n, feature_variance=self.feature_variance, 
                                        max_features=self.max_features, min_features=self.min_features)
        
        preds_ = self.prediction_fn(layer)
        if self.target_other:
            return layer[np.where(preds_ != self.target_class)]
        return layer[np.where(preds_ == self.target_class)]
    
    
    def feature_selection(self, counterfactual):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. 
        Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class
        Inputs:
        counterfactual: e*
        """
        if self.verbose == True:
            print("Feature selection...")

        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete)), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        out = counterfactual.copy()
        reduced = 0
        
        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete[k]
            if self.target_other:
                if self.prediction_fn(new_enn.reshape(1, -1)) != self.target_class:
                    out[k] = new_enn[k]
                    reduced += 1
            else:
                if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                    out[k] = new_enn[k]
                    reduced += 1
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out
    
