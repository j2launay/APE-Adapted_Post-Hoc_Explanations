#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
from random import randrange, randint, choices, uniform, random
from scipy.stats import multinomial, chi2
import scipy as sp
import pandas as pd
import math

def distances(x, y, ape, metrics='w_euclidian', dataset=None):
    
    if metrics == 'mahalanobis':
        distance = 0
        if ape.max_mahalanobis == None:
            df_x = pd.DataFrame(x, columns=ape.feature_names)
            df = pd.DataFrame(x, columns=ape.feature_names)
            df_x['mahalanobis'] = mahalanobis(x=df_x, data=df[ape.feature_names], max_mahalanobis=ape.max_mahalanobis)
            df_x['p'] = 1 - chi2.cdf(df_x['mahalanobis'], 3)
            return df_x
        else:
            #df_x = pd.DataFrame([x, y], columns=ape.feature_names)
            if dataset is not None:
                df = pd.DataFrame(y, columns=ape.feature_names)
            else:
                df = pd.DataFrame(ape.test_data, columns=ape.feature_names)
            df_x = pd.DataFrame([x], columns=ape.feature_names)
            df_x['mahalanobis'] = mahalanobis(x=df_x, data=df[ape.feature_names], max_mahalanobis=ape.max_mahalanobis)
            df_x['p'] = 1 - chi2.cdf(df_x['mahalanobis'], 3)
            """df_y = pd.DataFrame([y], columns=ape.feature_names)
            df_y['mahalanobis'] = mahalanobis(x=df_y, data=df[ape.feature_names], max_mahalanobis=ape.max_mahalanobis)
            df_y['p'] = 1 - chi2.cdf(df_y['mahalanobis'], 3)
            return abs(df_x['mahalanobis'].item()-df_y['mahalanobis'].item())"""
            return df_x['mahalanobis'].item()
        #calculate p-value for each mahalanobis distance 
        
    continuous_features = [x for x in set(range(len(x))).difference(ape.categorical_features)]
    x_categorical, y_categorical = x[ape.categorical_features], y[ape.categorical_features]
    x_continuous, y_continuous = x[continuous_features], y[continuous_features]
    # We divide by 2 since the categorical data must have been one hot encoded before
    same_coordinates_categorical = (x_categorical.shape[0] - sum(x_categorical == y_categorical)) /2
    
    distance = 0
    for nb_feature in range(x_continuous.shape[0]):
        distance += abs(x_continuous[nb_feature] - y_continuous[nb_feature])/(ape.max_features[nb_feature] - ape.min_features[nb_feature])
    distance = distance + same_coordinates_categorical
    distance = distance/x.shape[0]

    if metrics == 'w_manhattan':
        distance = 0
        for nb_feature in range(x_continuous.shape[0]):
            distance += abs(x_continuous[nb_feature] - y_continuous[nb_feature])/(ape.max_features[nb_feature] - ape.min_features[nb_feature])
        distance = distance + same_coordinates_categorical
        distance = distance/x.shape[0]
    else:
        distance = 0
        for nb_feature in range(x_continuous.shape[0]):
            temp_distance = ((x_continuous[nb_feature] - ape.mean_features[nb_feature]) - (y_continuous[nb_feature] - ape.mean_features[nb_feature]))\
                / ape.feature_variance[continuous_features[nb_feature]]
            distance += temp_distance * temp_distance
        distance = math.sqrt(distance)
        distance += same_coordinates_categorical
        try:
            distance = distance/ape.farthest_distance
        except AttributeError:
            # Distance to the farthest counterfactual has not been yet calculated
            pass
        except TypeError:
            # We take a new instance and the maximum distance must be computed again
            pass
        except ZeroDivisionError:
            pass
    
    return distance

def mahalanobis(x=None, data=None, cov=None, max_mahalanobis=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    if max_mahalanobis != None:
        mahal /= max_mahalanobis
    return mahal.diagonal()

def get_distances(x1, x2, metrics=None, categorical_features=[]):
    """
    Function that computes the distance between x1 and x2 based on euclidean metric or l0 norm (sparsity)
    Args: x1, x2: instances for which we want to compute the distance
          metrics: 'euclidean' or 'sparsity', variable to select the distance metric used
          categorical_features: List of features that are categorical
    Return: A dictionary of the distance between x1 and x2 
    """
    # Convert x1 and x2 to instance with only continuous features and instance with categorical feature in order to
    # measure a distance between the euclidean and the sparsity 
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

def generate_inside_ball(center, segment, n):
    """
    Args:
        "center" corresponds to the target instance to explain
        Segment corresponds to the size of the hypersphere
        n corresponds to the number of instances generated
        feature_variance: Array of variance for each continuous feature
    """
    def norm(v):
        # For Thibault Laugel
        v = np.linalg.norm(v, ord=2, axis=1)
        return v
    # Just for clarity of display
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    # Draw uniformaly instances between the value of s egment[0]**d and segment[1]**d with 
    # d the number of dimension of the instance to explain
    u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z

def generate_inside_field(center, segment, n, max_features, min_features, feature_variance):
    """
    Args:
        "center" corresponds to the target instance to explain
        Segment corresponds to the size of the hypersphere
        n corresponds to the number of instances generated
        feature_variance: Array of variance for each continuous feature
    """
    #print("segment", segment)
    if segment[0] == 1 and max_features == []:
        print("There is a problem since the radius is 1 and max feature has not been initialised", segment, center, feature_variance)
        generated_instances += 2
    d = center.shape[0]
    generated_instances = np.zeros((n,d))
    for feature, (min_feature, max_feature) in enumerate(zip(min_features, max_features)):
        range_feature = max_feature - min_feature
        # Modify the generation of artificial instance depending on the variance of each feature
        y = - segment[0] * range_feature
        z = segment[1] * range_feature
        variance = feature_variance[feature] * segment[1]
        k = (12 * variance)**0.5
        # Compute the minimum bound to generate instance inside field based on the variance
        a1 = min(y, z - k)
        b1 = a1 + k
        nb_instances = int (n / 2)
        generated_instances[:nb_instances, feature] = np.random.uniform(a1, b1, nb_instances)
        generated_instances[nb_instances:, feature] = np.random.uniform(-a1, -b1, n-nb_instances)
        np.random.shuffle(generated_instances[:,feature])
    generated_instances += center
    return generated_instances

def generate_categoric_inside_ball(center, segment, percentage_distribution, n, continuous_features, categorical_features, 
                            categorical_values, min_features, max_features, feature_variance, probability_categorical_feature=None, libfolding=False):
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
          libfolding: If set to True generate randomly instances and convert for categorical features 
                        the categorical values in probability distribution values
    Return: Matrix of n generated instances perturbed randomly around center in the area of the segment based on the data distribution  
    """
    def perturb_continuous_features(continuous_features, n, feature_variance, segment, center, matrix_perturb_instances):
        """
        Perturb each continuous features of the n instances around center in the area of a sphere of radius equals to segment
        Return a matrix of n instances of d dimension perturbed based on the distribution of the dataset
        """
        d = len(continuous_features)
        generated_instances = np.zeros((n,d))
        for feature, (min_feature, max_feature) in enumerate(zip(min_features, max_features)):
            range_feature = max_feature - min_feature
            # Modify the generation of artificial instance depending on the variance of each feature
            y = - segment[0] * range_feature
            z = segment[1] * range_feature
            variance = feature_variance[feature] * segment[1]
            k = (12 * variance)**0.5
            a1 = min(y, z - k)
            b1 = a1 + k
            nb_instances = int (n / 2)
            generated_instances[:nb_instances, feature] = np.random.uniform(a1, b1, nb_instances)
            generated_instances[nb_instances:, feature] = np.random.uniform(-a1, -b1, n-nb_instances)
            np.random.shuffle(generated_instances[:,feature])
        generated_instances += center[continuous_features]
        for nb, continuous in enumerate(continuous_features):
            matrix_perturb_instances[:,continuous] = generated_instances[:,nb].ravel()
        return matrix_perturb_instances
    
    if segment[1] > 1:
        print("There is a problem since the distance is superior to 1")
        print("radius", segment)
        segment = list(segment)
        segment[1] = 1
        segment = tuple(segment)

    matrix_perturb_instances = np.zeros((n, len(center)))
    for i in range(len(categorical_features)):
        # value_libfolding generates n instances between 0 and percentage distribution 
        # (probabilities values inferior to "percentage_distribution")
        value_libfolding = np.random.uniform(0, percentage_distribution, n)
        # add for each categorical feature these values to be considered as a probability 
        matrix_perturb_instances[:, categorical_features[i]] = value_libfolding

    if libfolding:
        matrix_perturb_instances_libfolding = matrix_perturb_instances.copy()
    for nb_categorical_features, categorical_feature in enumerate(categorical_features):
        value_target_instance = center[categorical_feature]
        for nb_instance, artificial_instance in enumerate(matrix_perturb_instances):
                # if a random number is superior to the probability of the categorical feature for artificial instance
                # we do not modify its value and kept the value of the target instance
                # otherwise we generate based on the probability of distribution from the dataset one trial
                # and store the corresponding categorical value 
                if random() < artificial_instance[categorical_feature]:
                    categorical_value = np.random.choice(categorical_values[nb_categorical_features], 1, p=probability_categorical_feature[nb_categorical_features])[0]
                    matrix_perturb_instances[nb_instance][categorical_feature] = categorical_value
                else:
                    matrix_perturb_instances[nb_instance][categorical_feature] = value_target_instance
    
    matrix_perturb_instances = perturb_continuous_features(continuous_features, n, feature_variance, 
                                                            segment, center, matrix_perturb_instances)
    if libfolding:
        matrix_perturb_instances_libfolding = perturb_continuous_features(continuous_features, n, 
                                                                feature_variance, segment, center, 
                                                                matrix_perturb_instances_libfolding)

    if libfolding:
        return matrix_perturb_instances, matrix_perturb_instances_libfolding
    else:
        return matrix_perturb_instances
