import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import re
import random
import os
import sys

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

def plot_tree(instances, labels, color, size=240, label=""):
    if label == "":
        plt.scatter(instances, labels, marker='s', s=size, color=color)
    else:
       plt.scatter(instances, labels, marker='s', s=size, color=color, label=label)

def plot_linear_boundary(instances, labels, color='b--', label='local Surrogate\nboundary'):
    plt.plot(instances, labels, color, label=label)
    return 0

def plot_subfigure(classif, X, Y, clf, x_min, x_max, y_min, y_max, target, x_sphere, y_sphere, x_tree, y_tree, linear_model, multiclass=False):
    if len(X[0]) > 2:
        X = PCA(n_components=2).fit_transform(X)

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    
    print("la génération d'instance va être longue....")
    x_test = np.linspace(min_x-3, max_x+3, 300) 
    y_test = np.linspace(min_y-3, max_y+3, 300)

    #classif = OneVsRestClassifier(SVC(kernel='linear'))
    #classif.fit(X, Y)

    plt.plot(225)
    
    value_to_print = []
    for value_x in x_test:
        for value_y in y_test:
            value_to_print.append([value_x, value_y])
    plot_labels = classif.predict(value_to_print)
    value_to_print = np.array(value_to_print)
    test_index_class_1 = np.where([y == 1 for y in plot_labels])
    test_index_class_2 = np.where([y == 0 for y in plot_labels])
    value_to_print_class1 = value_to_print[test_index_class_1]
    value_to_print_class2 = value_to_print[test_index_class_2]
    plot_tree(value_to_print_class2[:,0], value_to_print_class2[:,1], 'cyan', size=100, label='Boundary\nfor class 1\nfor Decision Tree')
    plot_tree(value_to_print_class1[:,0], value_to_print_class1[:,1], 'brown', size=100, label='Boundary\nfor class 2\nfor Decision Tree')

    # Plot the linear function for explanation
    labels_proba = linear_model.predict_proba(value_to_print)[:,0]
    index_proba = np.where([np.round(y, decimals=2) == 0.5 for y in labels_proba])
    value_linear_model = value_to_print[index_proba]
    plot_linear_boundary(value_linear_model[:,0], value_linear_model[:,1], label='Local Surrogate\nboundary')
    print("c'est bon pour le modèle linéaire de Lime")

    # Plot a random logistic regression to compare
    model_test = LogisticRegression()
    model_test.fit(value_to_print, plot_labels)
    labels_proba = model_test.predict_proba(value_to_print)[:,0]
    index_proba = np.where([np.round(y, decimals=2) == 0.5 for y in labels_proba])
    value_linear_model = value_to_print[index_proba]
    plot_linear_boundary(value_linear_model[:,0], value_linear_model[:,1], color='y--', label='Logistic Regression\nboundary')

    if multiclass:
        zero_class = np.where(Y == 0)
        one_class = np.where(Y == 1)
        two_class = np.where(Y == 2)
    else:
        zero_class = np.where(Y == 0)
        one_class = np.where(Y == 1)
    

    # TODO remove the x_tree and y_tree from input fonction and x_sphere, y_sphere
    # plot_tree(x_tree, y_tree, 'green', label='Boundary\nfor Decision Tree')
    
    #draw_linear_explanation(plt, x_sphere, y_sphere, clf, x_min, y_min, x_max, y_max, target, 
    #                                            clf.predict(target.reshape(1, -1)), len(set(Y)))
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=120, edgecolors='blue',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=160, edgecolors='yellow',
                facecolors='none', linewidths=2, label='Class 2')
    if multiclass:
        plt.scatter(X[two_class, 0], X[two_class, 1], s=120, edgecolors='green',
                facecolors='none', linewidths=2, label='Class 3')
    #plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
    #                'Boundary\nfor class 1')
    
    if multiclass:
        plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k:',
                    'Boundary\nfor class 2')
        plot_hyperplane(classif.estimators_[2], min_x, max_x, 'k.-',
                    'Boundary\nfor class 3')
        
    """
    elif dataset == "multilabel":
        plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    """
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')

def final_plot(cf_list_one, cf_list_other, cluster_centers_, counterfactual_in_sphere, instances_in_sphere, multiclass, filename):
    plt.scatter(cf_list_other[:, 0], cf_list_other[:, 1], marker='D', edgecolors='k', alpha=0.9, color='green', label='counterfactual')
    
    # TODO remettre ces 2 lignes ?
    plt.scatter(instances_in_sphere[:, 0], instances_in_sphere[:, 1], s=60, marker='+', edgecolors='k', 
                                                    alpha=0.5, c='black', label='instance generated\ninside sphere')
    plt.scatter(counterfactual_in_sphere[:, 0], counterfactual_in_sphere[:, 1], s=60, marker='+', edgecolors='k', alpha=0.5, 
                                                    c='white', label='counterfactual in\nsphere generated\ninside sphere')
    
    plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1], marker='v', edgecolors='k', alpha=1, color='olive', label='clusters center')
    if multiclass:
        plt.scatter(cf_list_one[:, 0], cf_list_one[:, 1], marker='D', edgecolors='k', alpha=0.9, color='pink', label='counterfactual from class 2')
    plt.title('Target instances (red) and their generated counterfactuals (green / blue / yellow)')
    plt.tight_layout()
    plt.legend(loc="upper left")
    os.makedirs(os.path.dirname(filename+"/"), exist_ok=True)
    plt.savefig(filename+"graph.png")
    plt.show()
    return plt

def pick_anchors_informations(anchors, x_min=-10, width=20, y_min=-10, height=20, compute=False):
    """
    Function to store information about the anchors and return the position and size to draw the anchors
    Anchors is of the form : "2 < x <= 7, -5 > y" or any rule
    """
    regex = re.compile(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    if len(anchors) == 0:
        return x_min, y_min, width, height
    elif len(anchors) == 1:
        if "x" in anchors[0]:
            x_bounds = regex.findall(anchors[0])
            x_min = min(float(x) for x in x_bounds)
            if len(x_bounds) == 1:
                width = (-width if "<" in anchors[0] else width)
            else:
                width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
        else:
            y_bounds = regex.findall(anchors[0])
            y_min = min(float(y) for y in y_bounds)
            if len(y_bounds) == 1:
                height = (-height if "<" in anchors[0] else height)
            else:
                height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)

    else:
        if "x" in anchors[0]:
            x_bounds = regex.findall(anchors[0])
            x_min = min(float(x) for x in x_bounds)
            if len(x_bounds) == 1:
                width = (-width if "<" in anchors[0] else width)
            else:
                width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
            y_bounds = regex.findall(anchors[1])
            y_min = min(float(y) for y in y_bounds)
            if len(y_bounds) == 1:
                height = (-height if "<" in anchors[1] else height)
            else:
                height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)
        else:
            y_bounds = regex.findall(anchors[0])
            y_min = min(float(y) for y in y_bounds)
            if len(y_bounds) == 1:
                height = (-height if "<" in anchors[0] else height)
            else:
                height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)
            x_bounds = regex.findall(anchors[1])
            x_min = min(float(x) for x in x_bounds)
            if len(x_bounds) == 1:
                width = (-width if "<" in anchors[1] else width)
                print(width)
            else:
                width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
    return x_min, y_min, width, height

def draw_rectangle(ax, x_min_anchors, y_min_anchors, width, height, cnt):
    """
    Draw the rectangle upon the graphics
    """
    if y_min_anchors != -10:#y_min-4:
        ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-', color='grey', label='anchor border') if cnt == 1 else ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-', color='grey')
        ax.plot([x_min_anchors + width, x_min_anchors], [y_min_anchors + height, y_min_anchors + height], 'y-', color='grey')
    else:
        ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-', linestyle='dashed', color='grey', label='anchor border') if cnt == 1 else ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-', color='grey')
        ax.plot([x_min_anchors + width, x_min_anchors], [y_min_anchors + height, y_min_anchors + height], 'y-', linestyle='dashed', color='grey')
    if x_min_anchors != -10:#x_min-4 :
        ax.plot([x_min_anchors, x_min_anchors], [y_min_anchors, y_min_anchors + height], 'g-', color='grey')
        ax.plot([x_min_anchors + width, x_min_anchors + width], [y_min_anchors, y_min_anchors + height], 'b-', color='grey')
    else:
        ax.plot([x_min_anchors, x_min_anchors], [y_min_anchors, y_min_anchors + height], 'g-', linestyle='dashed', color='grey')
        ax.plot([x_min_anchors + width, x_min_anchors + width], [y_min_anchors, y_min_anchors + height], 'b-', linestyle='dashed', color='grey')

def draw_linear_explanation(ax, x_sphere, y_sphere, clf, x_min, y_min, x_max, y_max, target, target_class, nb_class) :
    # Fit the classifier
    #y_score = clf.predict(x_sphere)
    #roc_auc = roc_auc_score(y_true=y_sphere, y_score=y_score, multi_class='ovo') if nb_class > 2 else roc_auc_score(y_sphere, y_score)
    #print("courbe roc", roc_auc)
    X, Y, weights = generate_neighbors(target[0], target[1], clf, x_min, x_max,
                       y_min, y_max, n=100, x_avg=None, y_avg=None, sd=None)
    clf = LinearRegression()#LogisticRegression()#LinearRegression()
    clf.fit(X, Y, weights)

    x_test = np.linspace(x_min, x_max, 100) 
    y_test = np.linspace(y_min, y_max, 100)
    x_prime = np.transpose([np.tile(x_test, len(y_test)), 
                           np.repeat(y_test, len(x_test))])

    x_test, y_test = np.meshgrid(x_test, y_test)

    loss = (np.matmul(x_prime, np.transpose(clf.coef_)) + clf.intercept_).ravel()#clf.predict_proba(x_prime)#
    print("TEST LOSS", loss)
    min_loss = min(loss)
    max_loss = max(loss)
    a_max = max_loss
    standard_deviation_print = 0.02
    if nb_class > 2:
        if target_class == 0:
            a_min = (max_loss+min_loss)/nb_class - standard_deviation_print
        elif target_class == nb_class-1:
            a_min = (target_class)*(max_loss+min_loss)/nb_class - standard_deviation_print
        else:
            a_min = (target_class)*(max_loss+min_loss)/nb_class
            a_max = (target_class + 1)*(max_loss+min_loss)/nb_class
    else:
        a_min = (max_loss+min_loss)/2 - standard_deviation_print
    if a_max == max_loss : 
        a_max = a_min + 2 * standard_deviation_print

    loss = np.clip(loss, a_min=a_min, a_max=a_max)
    loss =  np.array(np.split(loss, 100))
    ax.pcolormesh(x_test, y_test, loss, cmap='gray')

    ax.axis([x_min, x_max, y_min, y_max])
    return clf#, roc_auc    

def generate_neighbors(target_instance_x, target_instance_y, clf, x_min, x_max,
                       y_min, y_max, n=100, x_avg=None, y_avg=None, sd=None) :
    points = []
    x_delta = x_max - x_min   
    y_delta = y_max - y_min
    distances = np.ones(n)
    farther_distances = np.full((n, ), 2.)

    ## Add the target
    for i in range(n) :
        points.append((target_instance_x, target_instance_y))
    
    if x_avg is None or y_avg is None :
        for i in range(int(n/2)) :
            points.append((target_instance_x, y_min
                           + random.random() * y_delta ))
    
        for i in range(int(n/2)) :
            points.append((x_min + random.random() * x_delta ,
                           target_instance_y))
            
        for i in range(n) :
            points.append((x_min + random.random() * x_delta,
                           y_min + random.random() * y_delta))
    
    X = np.array(points)
    Y = clf.predict(X)

    return X, Y, np.concatenate((np.zeros(n), distances, farther_distances))
