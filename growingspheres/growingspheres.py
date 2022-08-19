#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils.gs_utils import generate_inside_ball, get_distances
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state



class GrowingSpheres:
    """
    class to fit the Original Growing Spheres algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                n_in_layer=2000,
                first_radius=0.1,
                dicrease_radius=10,
                sparse=True,
                verbose=False,
                continuous_features=None,
                categorical_features=[],
                categorical_values=[],
                feature_variance=None,
                farthest_distance_training_dataset=None,
                probability_categorical_feature=None,
                min_counterfactual_in_sphere=0,
                max_features=None,
                min_features=None):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        
        if target_class == None:
            target_class = 1 - self.y_obs
        
        self.target_class = target_class
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius 
        self.sparse = sparse
        
        # For experiments to compare with Growing Fields on dataset with categorical features
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.verbose = verbose
        
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

        
    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        ennemies_, radius = self.exploration()
        ennemies = sorted(ennemies_, 
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))
        self.e_star = ennemies[0]
        if self.sparse == True:
            out = self.feature_selection(ennemies[0])
        else:
            out = ennemies[0]
        return out, ennemies, radius
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_ = 999
        radius_ = self.first_radius
        
        while n_ennemies_ > 0:
            first_layer_ = self.ennemies_in_layer_((0, radius_), self.n_in_layer)
            n_ennemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.dicrease_radius
            if self.verbose == True:
                print("%d ennemies found in initial sphere. Zooming in..."%n_ennemies_)
            
        else:
            if self.verbose == True:
                print("Exploring...")
            iteration = 0
            step_ = (self.dicrease_radius - 1) * radius_/5.0
            
            while n_ennemies_ <= 0:
                layer = self.ennemies_in_layer_((radius_, radius_ + step_), self.n_in_layer)
                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                iteration += 1
            if self.verbose == True:
                print("Final number of iterations: ", iteration)
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return layer, radius_
    
    
    def ennemies_in_layer_(self, segment, n=1000):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        layer = self.generate_inside_spheres(self.obs_to_interprete, segment, n)
        preds_ = self.prediction_fn(layer)
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
            if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                out[k] = new_enn[k]
                reduced += 1
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out

    def generate_inside_spheres(self, center, segment, n, feature_variance=None):
        """
        Args:
            "center" corresponds to the target instance to explain
            Segment corresponds to the size of the hypersphere
            n corresponds to the number of instances generated
            feature_variance: Array of variance for each continuous feature
        """
        def norm(v):
            v= np.linalg.norm(v, ord=2, axis=1)
            return v
            
        # Just for clarity of display
        d = center.shape[0]
        z = np.random.normal(0, 1, (n, d))
        # Draw uniformaly instances between the value of segment[0]**d and segment[1]**d with d the number of dimension of the instance to explain
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
        r = u**(1/float(d))
        z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
        z = z + center
        return z