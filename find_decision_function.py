import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
# models and dataset import
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from yellowbrick.cluster import KElbowVisualizer
# GS and our code import
from growingspheres import counterfactuals as cf
from growingspheres.utils.gs_utils import get_distances, generate_inside_field, distances
from generate_dataset import generate_dataset
from plot_functions import plot_hyperplane, plot_subfigure, final_plot, pick_anchors_informations, draw_rectangle, draw_linear_explanation
from anchors import utils, anchor_tabular, anchor_base
from lime.lime_text import LimeTextExplainer
from anchors import limes
import pyfolding as pf
import baseGraph
import spacy
import os
import ape_tabular
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
PATH = ''

def get_farthest_distance(train_data, target_instance, ape):
    farthest_distance = 0
    for training_instance in train_data:
        # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
        # but it adds a sparsity distance computation (i.e: number of same values) 
        #farthest_distance_now = get_distances(training_instance, instance, categorical_features=self.categorical_features)["euclidean"]
        farthest_distance_now = distances(training_instance, target_instance, ape)
        if farthest_distance_now > farthest_distance:
            farthest_distance = farthest_distance_now
    return farthest_distance

def convert_raw_data(obs, dataset_name):
    # Convert a vectorize sentence into a raw text data
    if dataset_name == "multilabel": obs = obs.reshape(1, -1)
    return obs

def compute_distance(point, clf):
    # Compute the distance from a decision frontier for a point
    try:
        distance = clf.predict_proba([point])
    except AttributeError:
        distance = clf.decision_function([point])
    return distance

def instance_around_multiple_classes(smallest_zero, largest_zero, smallest_two, largest_two):
    # Return True if the mean of the smallest and largest value from a class is contains 
    # in the range of the smallest and the largest distance of a different class
    return smallest_zero < ((largest_two + smallest_two)/2) < largest_zero or smallest_two < ((largest_zero + smallest_zero)/2) < largest_two

def get_points_from_class(classe, x_data, clf, dataset=None):
    # Return each points from X that belongs to the class classe following the prediction of clf
    try:
        return x_data[np.where(clf.predict(x_data) == classe)], np.where(clf.predict(x_data) == classe)[0]
    except:
        return x_data[np.where(clf.predict(x_data)[:, classe])], np.where(clf.predict(x_data) == classe)[0]

def get_growing_sphere_from_class(opponent_class, growing_sphere_enemies, raw_data, clf, 
                                continuous_features, categorical_features, categorical_values, obs, ape):
    growing_sphere = cf.CounterfactualExplanation(raw_data, clf.predict, method='GF', target_class=opponent_class, 
                continuous_features=continuous_features, categorical_features=categorical_features, categorical_values=categorical_values,
                max_features=ape.max_features, min_features=ape.min_features) 
    growing_sphere.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=True, feature_variance=ape.feature_variance)
    final_largest = distances(growing_sphere.onevsrest[-1], raw_data, ape)
    final_smallest = distances(growing_sphere.onevsrest[0], raw_data, ape)
    for instance_test in growing_sphere.onevsrest:
        distance_counterfactual = distances(instance_test, raw_data, ape)
        if distance_counterfactual > final_largest:
            print("yeah we expand distance from", final_largest, "to", distance_counterfactual, "distance")
            final_largest = distance_counterfactual
        elif distance_counterfactual < final_smallest:
            print("Ooh we found smallest counterfactual", final_smallest, "from", distance_counterfactual)
            final_smallest = distance_counterfactual

    #largest = get_distances(growing_sphere.onevsrest[-1], obs)["euclidean"]
    #smallest = get_distances(growing_sphere.onevsrest[0], obs)["euclidean"]
    if opponent_class == None:
        print("classe la plus proche : ", clf.predict(growing_sphere.onevsrest[0].reshape(1, -1)))
    print("largest distance from class ", opponent_class, " : " , final_largest)
    print("smallest distance from class ", opponent_class, " : " , final_smallest)
    return growing_sphere.enemy, final_largest, final_smallest, growing_sphere.onevsrest, growing_sphere.radius

def preparing_dataset(x, y, plot, dataset_name, model):
    if plot:
        x = PCA(n_components=2).fit_transform(x)
    dataset = utils.load_dataset(dataset_name, balance=True, discretize=False, dataset_folder="./dataset", X=x, y=y, plot=plot)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    return dataset, model, x_train, x_test, y_train, y_test

def plot_explanation_results(X, cnt, x_test, y_test, multiclass, model, obs, x_in_sphere, 
                            y_in_sphere, anchor_exp, plt, x_tree, y_tree, 
                            linear_model, plot=True, instances_x_test=[]):
        x_min = min([x[0] for x in X])
        x_max = max([x[0] for x in X])
        y_min = min([y[1] for y in X])
        y_max = max([y[1] for y in X])
        """
        # Informations sur le modèle linéaire appris par Lime
        x_plot = np.linspace([x_min-5, y_min -5], [x_max+5, y_max +5], 10)
        y_plot = lime_exp.easy_model.predict(x_plot)
        print("ordonnées à l'origine ", lime_exp.intercept)
        print("coef", lime_exp.as_list(label=target_class))
        coef_ = lime_exp.as_list(label=target_class)[0][1] * x_plot + lime_exp.intercept[target_class]
        """
        #if cnt < 1 :
        (None, plot_subfigure(model, x_test, y_test, x_tree=x_tree, y_tree=y_tree, multiclass=multiclass, clf=model, x_min=x_min-8, y_min=y_min-8, x_max=x_max+8, 
                                        y_max=y_max+8, target=obs, x_sphere=x_in_sphere, y_sphere=y_in_sphere, linear_model=linear_model))
        """
        else:
            auc_score = draw_linear_explanation(plt, x_in_sphere, y_in_sphere, model, x_min, y_min, x_max, y_max, obs, 
                                                    model.predict(obs.reshape(1, -1)), len(set(y_test)))
        """
        #plt.scatter(instances_x_test[:,0], instances_x_test[:,1], marker='*', s=260, color='green', label='instances radius 1')            
        plt.scatter(obs[0],obs[1], marker='*', s=260, color='yellow', label='target instance')            
        x_min_anchors, y_min_anchors, width, height = pick_anchors_informations(anchor_exp.names(), x_min=x_min-4, 
                                        y_min=y_min-4, width=x_max-x_min+8, height=y_max-y_min+8)
        draw_rectangle(plt, x_min_anchors, y_min_anchors, width, height, cnt)


def return_instance_in_cf_growing_sphere(x_instances_to_test_if_in_sphere, obs, closest_counterfactual, longest_distance_other, ape):#, growing_sphere_zero_enemies):
    # Compute the shortest distance and return instance in the area of an hyper sphere of radius equals to the minimal distance
    # of a counterfactual instance to the target instance
    #longest_distance_other = pairwise_distances(obs.reshape(1, -1), growing_sphere_other_enemies[-1].reshape(1, -1))[0][0]
    #longest_distance_zero = pairwise_distances(obs.reshape(1, -1), growing_sphere_zero_enemies[-1].reshape(1, -1))[0][0]
    #counterfactual_explanation = min(longest_distance_other, longest_distance_zero)
    counterfactual_explanation = longest_distance_other
    print("l'ennemi le plus loin est a une distance de : ", counterfactual_explanation)
    print("La cible est : ", obs, " l'instance contrefactuelle est : ", closest_counterfactual)
    position_instances_in_sphere = []
    nb_instance_in_sphere = 0
    for position, i in enumerate(x_instances_to_test_if_in_sphere):
        #if pairwise_distances(closest_counterfactual.reshape(1, -1), i.reshape(1, -1))[0] < counterfactual_explanation:
        if distances(i, obs, ape) < counterfactual_explanation:
            position_instances_in_sphere.append(position)
            nb_instance_in_sphere += 1
    print("nb instances dans la sphere ", nb_instance_in_sphere)
    nb_instance_in_sphere = 100 if nb_instance_in_sphere == 0 else nb_instance_in_sphere
    return position_instances_in_sphere, nb_instance_in_sphere

def minimum_instance_in_sphere(nb_min_instance_in_sphere, nb_instance_in_sphere, closest_counterfactual, radius_enemies, percentage_instance_sphere, 
                            x_train, position_instances_in_sphere, target_class, model, ape):
    nb_different_outcome = 0
    while nb_different_outcome < nb_min_instance_in_sphere: 
        nb_instance_in_sphere = 2*nb_instance_in_sphere
        generated_instances_inside_sphere = generate_inside_field(closest_counterfactual, (0, radius_enemies), 
                                                    (int) (nb_instance_in_sphere*2*percentage_instance_sphere),
                                                    ape.max_features, ape.min_features, ape.feature_variance)
        x_in_sphere = np.append(x_train[position_instances_in_sphere], generated_instances_inside_sphere, axis=0) if position_instances_in_sphere != [] else generated_instances_inside_sphere
        y_in_sphere = model.predict(x_in_sphere)#y[position_instances_in_sphere]
        #print('il y a ', nb_instance_in_sphere, " instances dans la sphère sur ", len(x_train), " instances au total.")
            
        for y_sphere in y_in_sphere:
            if y_sphere != target_class:
                nb_different_outcome += 1
    print('il y a ', nb_different_outcome, " instances d'une classe différente dans la sphère sur ", len(x_in_sphere))
    return x_in_sphere, y_in_sphere


if __name__ == "__main__":
    plot = True
    verbose = True
    nb_instance = 10
    nb_min_instance_in_sphere = 20
    threshold_interpretability = 0.95 # Threshold for minimum accuracy of Lime and Anchors
    percentage_instance_sphere = 0.1 # less corresponds to less artificial instances generated inside the sphere.
    dataset_name = "generate_moons" # generate_blobs, generate_moons, iris, adult, titanic, blood
    models = [OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)),
                tree.DecisionTreeClassifier(), svm.SVC(kernel='linear', probability=True), MLPClassifier(random_state=1, max_iter=300),  
                LogisticRegression(),
                GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1)]    
    models = [tree.DecisionTreeClassifier(max_depth=3), LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(n_estimators=50)]
    for nb_model, model in enumerate(models):
        model_name = type(model).__name__
        print("LE MODELE UTILISE EST : ", model_name)
        X, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names = generate_dataset(dataset_name)
        dataset, model, x_train, x_test, y_train, y_test = preparing_dataset(X, y, plot, dataset_name, model)
        """
        anchor_explainer, lime_explainer, c, model, x_test_class, x_raw_text, predict_fn, predict_fn_lime = compute_model_and_interpretability_models(regression, 
                                                                                    class_names, model, dataset, x_train, y_train, 
                                                                                    x_test, y_test, target_class, dataset_name)
        """
        model = model.fit(x_train, y_train)
        print(' ### Accuracy:', model.score(x_test, y_test)) if regression else print(' ### Accuracy:', sum(model.predict(x_test) == y_test)/len(y_test))
        # Initialize for plot
        """cf_list_zero = [] 
        cf_list_two = []""" 
        counterfactual_instances = []
        for cnt, obs in enumerate(x_test):
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, model.predict, #black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=True, threshold_precision=threshold_interpretability)
            neigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=distances, metric_params={"ape": explainer})
            #neigh.fit([x_test, x_train, explainer.max_features, explainer.min_features], y_test)
            neigh.fit(x_test, y_test)
            print("TEST", neigh.kneighbors(x_test))
            precision, coverage, f2, multimodal_result = explainer.explain_instance(obs, 
                                                            all_explanations_model=True)
            


            target_class = model.predict(obs.reshape(1, -1))[0]
            explainer.target_class = target_class
            if cnt == nb_instance:
                break
            print('====================================================', cnt)
            raw_data = convert_raw_data(obs, dataset_name)
            print("observation ", raw_data)

            print("Try to find closest boundary ")
            growing_sphere = cf.CounterfactualExplanation(raw_data, model.predict, method='GF', target_class=None, 
                                                    continuous_features=continuous_features, categorical_features=categorical_features, 
                                                    categorical_values=categorical_values, max_features=explainer.max_features,
                                                    min_features=explainer.min_features) 
            growing_sphere.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=True,
                                feature_variance=explainer.feature_variance)
            farthest_distance = get_farthest_distance(x_train, explainer.closest_counterfactual, explainer)
            position_instances_in_sphere, nb_training_instance_in_sphere = explainer.instances_from_dataset_inside_sphere(explainer.closest_counterfactual, 
                                                                                                                explainer.extended_radius, x_train)
            print("Generate instances in the area of a sphere around the counter factual instance")
            instances_in_sphere, _, _, _ = explainer.generate_instances_inside_sphere(explainer.extended_radius, 
                                                    explainer.closest_counterfactual,  x_test, farthest_distance, 
                                                    explainer.nb_min_instance_per_class_in_sphere,
                                                    [], 0)
            counterfactual_instances_in_sphere = np.array(explainer.store_counterfactual_instances_in_sphere(instances_in_sphere, explainer.target_class))
            print("done generating")
            print()
            print()

            x_instances_to_test_if_in_sphere = x_train
            position_instances_in_sphere, nb_instance_in_sphere = return_instance_in_cf_growing_sphere(x_instances_to_test_if_in_sphere, obs,
                                                                                    growing_sphere.enemy, farthest_distance, explainer)

            if explainer.target_class != target_class:
                print()
                print()
                print("il y a un problème dans les classes cibles")
                print()
                print
            x_in_sphere, y_in_sphere = minimum_instance_in_sphere(nb_min_instance_in_sphere, nb_instance_in_sphere, 
                            growing_sphere.enemy, growing_sphere.radius, 
                            percentage_instance_sphere, x_train, position_instances_in_sphere, 
                            explainer.target_class, model, explainer)
            
            # Le hic est sur le nombre de counterfactual instances
            print("IL Y A ", len(counterfactual_instances_in_sphere), " ENEMIES over", len(instances_in_sphere))
            results = pf.FTU(counterfactual_instances_in_sphere, routine="python")
            print(results)
            
            multimodal_results = results.folding_statistics<1
            #if multimodal_results:
            visualizer = KElbowVisualizer(KMeans(), k=(1,8))
            x_elbow = np.array(counterfactual_instances_in_sphere)
            visualizer.fit(x_elbow)
            n_clusters = visualizer.elbow_value_
            if n_clusters is not None:
                if verbose: print("n CLUSTERS ", n_clusters)
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(counterfactual_instances_in_sphere)
                clusters_centers = kmeans.cluster_centers_
                if verbose: print("Mean center of clusters from KMEANS ", clusters_centers)
            plt.show(block=False)
            plt.show(block=False)
            plt.pause(1)
            plt.close('all')

            print("searching for explanations...")
            anchor_exp = explainer.anchor_explainer.explain_instance(obs, model.predict, threshold=threshold_interpretability, 
                                        delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, 
                                        stop_on_first=False, desired_label=None, beam_size=4)
            print("anchors explanation find, now let's go for Lime !")
            lime_exp = explainer.lime_explainer.explain_instance_training_dataset(explainer.closest_counterfactual, model.predict_proba, 
                                    num_features=4, model_regressor=LogisticRegression(),
                                    instances_in_sphere=instances_in_sphere)
            linear_model = lime_exp.easy_model
            print("la précision de Anchors est de ", anchor_exp.precision())
            print('Lime explanation')
            print('\n'.join(map(str, lime_exp.as_list())))
            print('Anchor: %s' % (' AND '.join(anchor_exp.names())))

            print("predict observation; ",model.predict(obs.reshape(1, -1)))

            for i, j, k, l in zip (precision, f2, coverage, ['LS extend', 'APE', 'anchor']):
                print()
                if i == 0:
                    print("YEAH")
                elif i == 0.0:
                    print("okay")
                """
                if i == 0 and ("Tree" in model_name or "random" in model_name):
                    print()
                    tree.plot_tree(model)
                """
                print("precision of", l, i)
                print("f2 of", l, j)
                print("coverage of", l, k)
            # Changing type of data return by growing sphere to be numpy for ploting
            #cf_list_zero = np.array(growing_sphere_zero_enemies)
            cf_list_two = np.array([])#growing_sphere_two_enemies) if multiclass else None
            cf_list_other = np.array(growing_sphere.onevsrest)
            filename = os.getcwd() + "/results/"+ dataset_name+ "/" +type(model).__name__+ "/"
            print("filename", filename)
            #if precision[1] < precision[0] or precision[1] < precision[2]:
            y_in_sphere = model.predict(instances_in_sphere)
            x_in_sphere = instances_in_sphere[np.where([y == explainer.target_class for y in y_in_sphere])]
            x_sphere, y_sphere = [], []
            for element in x_in_sphere:
                x_sphere.append(element[0])
                y_sphere.append(element[1])
            
            """instances_in_sphere_test, _, _, _ = explainer.generate_instances_inside_sphere(1, 
                                                    explainer.closest_counterfactual,  x_test, farthest_distance, 
                                                    explainer.nb_min_instance_per_class_in_sphere,
                                                    [], 0)
            """
            print_x_test, print_x_train, print_y_test, print_y_train = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
            if coverage[2] == 1:
                print("La couverture est de 1 donc je regarde si il faut changer le test split, c'est pour vérifier que toutes les instances sur lesquelles je calcule la couverture")
                plot_explanation_results(X, cnt, print_x_test, print_y_test, multiclass, 
                                                model, obs, counterfactual_instances_in_sphere, 
                                                [explainer.target_class]*len(counterfactual_instances_in_sphere), anchor_exp, 
                                                plt, x_sphere, y_sphere, linear_model, plot, instances_in_sphere_test)
                final_plot(cf_list_two, cf_list_other, kmeans.cluster_centers_, counterfactual_instances_in_sphere, 
                                    instances_in_sphere, multiclass, filename)
            #elif precision[1] < precision [0] or precision[1] < precision[2]:
            #    print("ca me saoule")
            else:
                plot_explanation_results(X, cnt, print_x_train, print_y_train, multiclass, 
                                                model, obs, counterfactual_instances_in_sphere, 
                                                [explainer.target_class]*len(counterfactual_instances_in_sphere), anchor_exp, 
                                                plt, x_sphere, y_sphere, linear_model, plot)#, instances_in_sphere_test)
                final_plot(cf_list_two, cf_list_other, kmeans.cluster_centers_, counterfactual_instances_in_sphere, 
                                    instances_in_sphere, multiclass, filename)
                # TODO verifier que l'on ne fait pas de l'overfitting en regardant si c'est différent
            
                plot_explanation_results(X, cnt, print_x_test, print_y_test, multiclass, 
                                                model, obs, counterfactual_instances_in_sphere, 
                                                [explainer.target_class]*len(counterfactual_instances_in_sphere), anchor_exp, 
                                                plt, x_sphere, y_sphere, linear_model,plot)#, instances_in_sphere_test)
                final_plot(cf_list_two, cf_list_other, kmeans.cluster_centers_, counterfactual_instances_in_sphere, 
                                    instances_in_sphere, multiclass, filename)
            