#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
from random import randrange, randint, choices, uniform, random
from scipy.stats import multinomial

def distances(x, y, train_data, max_feature, min_feature, categorical_features=[]):
    continuous_features = [x for x in set(range(len(x))).difference(categorical_features)]
    x_categorical, y_categorical = x[categorical_features], y[categorical_features]
    x_continuous, y_continuous = x[continuous_features], y[continuous_features]
    #print("x", x)
    #print("y", y)
    same_coordinates_categorical = x_categorical.shape[0] - sum(x_categorical == y_categorical)
    distance = 0
    for nb_feature in range(x_continuous.shape[0]):
        distance += abs(x_continuous[nb_feature] - y_continuous[nb_feature])/(max_feature[nb_feature] - min_feature[nb_feature])
    distance = distance + same_coordinates_categorical
    distance = distance/x.shape[0]
    return distance

def get_distances(x1, x2, metrics=None, categorical_features=[]):
    """
    Function that computes the distance between x1 and x2 based on euclidean metric or l0 norm (sparsity)
    Args: x1, x2: instances for which we want to compute the distance
          metrics: 'euclidean' or 'sparsity', variable to select the distance metric used
          categorical_features: List of features that are categorical
    Return: A dictionary of the distance between x1 and x2 
    """
    # Convert x1 and x2 to instance with only continuous features and instance with categorical feature in order to measure a distance between the euclidean and the sparsity 
    continuous_features = [x for x in set(range(len(x1))).difference(categorical_features)]
    x1_categorical, x2_categorical = x1[categorical_features].reshape(1, -1), x2[categorical_features].reshape(1, -1)
    x1_continuous, x2_continuous = x1[continuous_features].reshape(1, -1), x2[continuous_features].reshape(1, -1)
    
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    # Compute the euclidean distance between x1 and x2
    euclidean = pairwise_distances(x1, x2)[0][0]
    # Compute the l0 norm distance between x1 and x2
    same_coordinates = sum((x1 == x2)[0])
    
    euclidean_continuous = pairwise_distances(x1_continuous, x2_continuous)[0][0]
    same_coordinates_categorical = x1_categorical.shape[1] - sum((x1_categorical == x2_categorical)[0])
    new_euclidean = euclidean_continuous+same_coordinates_categorical
    #pearson = pearsonr(x1, x2)[0]
    #kendall = kendalltau(x1, x2)
    out_dict = {'euclidean': new_euclidean, #euclidean,
                'sparsity': x1.shape[1] - same_coordinates#,
                #'kendall': kendall
               }
    return out_dict        

def generate_inside_ball(center, segment, n, feature_variance=None):
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
    z = np.zeros((n,d))
    if feature_variance is not None:
        for feature in range(d):
            # Modify the generation of artificial instance depending on the variance of each feature
            z[:,feature] = np.random.normal(0, feature_variance[feature], n)
    else:
        z = np.random.normal(0, 1, (n, d))
    # Draw uniformaly instances between the value of segment[0]**d and segment[1]**d with d the number of dimension of the instance to explain
    u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z

def generate_categoric_inside_ball(center, segment, percentage_distribution, n, continuous_features, categorical_features, 
                            categorical_values, feature_variance=None, probability_categorical_feature=None, libfolding=False):
    """
    Generate randomly instances inside a field based on the variance of each continuous feature and 
    the maximum of distribution probability of changing a value for categorical feature
    Args: center: instance centering the field
          segment: radius of the sphere (area in which instances are generated)
          percentage_distribution: Maximum distribution probability of changing the value of categorical features
          n: Number of instances generated in the field
          continuous_features: The list of features that are discrete/continuous
          categorical_features: The list of features that categorical
          categorical_values: Array of arrays containing the values for each categorical feature
          feature_variance: Array of variance for each continuous feature 
          probability_categorical_feature: Distribution probability for each categorical features of each values 
          libfolding: If set to True generate randomly instances and convert for categorical features the categorical values in probability distribution values
    Return: Matrix of n generated instances perturbed randomly around center in the area of the segment based on the data distribution  
    """
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)

    def perturb_continuous_features(continuous_features, n, feature_variance, segment, center, to_return):
        """
        Perturb each continuous features of the n instances around center in the area of a sphere of radius equals to segment
        Return a matrix of n instances of d dimension perturbed based on the distribution of the dataset
        """
        d = len(continuous_features)
        z = np.zeros((n,d))
        if feature_variance is not None:
            for feature in range(d):
                # Modify the generation of artificial instance depending on the variance of each feature
                z[:,feature] = np.random.normal(0, feature_variance[feature], n)
        else:
            z = np.random.normal(0, 1, (n, d))
        # Draw uniformaly instances between the value of segment[0]**d and segment[1]**d with d the number of dimension of the instance to explain
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
        r = u**(1/float(d))
        z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
        to_add = np.zeros((n, len(center)))
        for continuous in continuous_features:
            to_add[:,continuous] = center[continuous]
        z = z + to_add[:,continuous_features]
        for nb, continuous in enumerate(continuous_features):
            to_return[:,continuous] = z[:,nb].ravel()
        return to_return
    
    if segment[1] > 1:
        print("Il y a un problème puisque la distance est supérieure à 1")
    if percentage_distribution > 100:
        print("il y a un problème puisque le pourcentage de distribution est supérieur à 100")

    to_return = np.zeros((n, len(center)))
    for i in range(len(categorical_features)):
        # value_libfolding generates n instances between 0 and percentage distribution (probabilities values inferior to "percentage_distribution")
        value_libfolding = np.random.uniform(0, percentage_distribution, n)
        # add for each categorical feature these values to be considered as a probability 
        to_return[:, categorical_features[i]] = value_libfolding

    if libfolding:
        to_return_libfolding = to_return.copy()
    for nb_categorical_features, categorical_feature in enumerate(categorical_features):
        value_target_instance = center[categorical_feature]
        for nb_instance, artificial_instance in enumerate(to_return):
                # if a random number is superior to the probability of the categorical feature for artificial instance
                # we do not modify its value and kept the value of the target instance
                # otherwise we generate based on the probability of distribution from the dataset one trial
                # and store the corresponding categorical value 
                if random() < artificial_instance[categorical_feature]:
                    probability_repartition = multinomial.rvs(n=1, p=probability_categorical_feature[nb_categorical_features], size=1)[0]
                    categorical_value = categorical_values[nb_categorical_features][np.where(probability_repartition==1)[0][0]]
                    to_return[nb_instance][categorical_feature] = categorical_value
                else:
                    to_return[nb_instance][categorical_feature] = value_target_instance
    #np.set_printoptions(formatter={'float': '{:g}'.format})
    
    to_return = perturb_continuous_features(continuous_features, n, feature_variance, segment, center, to_return)
    if libfolding:
        to_return_libfolding = perturb_continuous_features(continuous_features, n, feature_variance, segment, center, to_return_libfolding)
    
    if libfolding:
        return to_return, to_return_libfolding
    else:
        return to_return
