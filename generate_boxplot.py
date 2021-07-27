import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import os
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential


"""
pyplot.subplot(121)
pyplot.boxplot([[1, 2, 3, 4, 5, 13], [6, 7, 8, 10, 10, 11, 12], [1, 2, 3]])
pyplot.ylim(0, 14)
pyplot.title('boxplot avec sequence')

pyplot.subplot(122)
pyplot.boxplot(np.array([[1, 2, 3], [2, 7, 8], [1, 3, 10], [2, 5, 12]]))
pyplot.ylim(0, 14)
pyplot.title('boxplot avec array 2d')

#pyplot.show()
"""

threshold_interpretability = "0.99"
datasets = ["generate_moons"]#["compas", "titanic", "adult"]#"generate_moons", "generate_blob", "generate_blobs", "artificial", "blood", "diabete", "iris",
models = [RandomForestClassifier(), LogisticRegression(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('dt', tree.DecisionTreeClassifier())], voting="soft"),
                #Sequential(),
                GradientBoostingClassifier(),
                tree.DecisionTreeClassifier(), 
                RidgeClassifier(), 
                MLPClassifier()]
graphs = ["coverage", "precision", "f1", 'recall']#, "average_distance", "lime_vs_ls"]

"""
for dataset in datasets:
    print(dataset)
    for model in models:
        model = type(model).__name__
        print(model)
        for graph in graphs:
            pyplot.subplot(212)
            filename = dataset + "/" + model + "/" + threshold_interpretability + "/" + graph
            data = pd.read_csv("./results/" + filename + ".csv")
            #print("data", data)
            pyplot.boxplot(data)
            pyplot.gca().xaxis.set_ticklabels(data.columns)
            pyplot.title('boxplot of ' + graph + " on " + dataset + " for " + model)
            os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
            pyplot.savefig("./boxplot/" + filename + ".png")
            pyplot.show(block=False)
            pyplot.pause(1)
            pyplot.close('all')
"""

# TODO modify for lime_ls to generate graph for each radius instead of a boxplot
graph_multiple = ["coverages", "precisions", "f1s", "lime_ls"]
for dataset in datasets:
    print(dataset)
    for graph in graph_multiple:
            pyplot.subplot(212)
            filename = dataset + "/" + threshold_interpretability + "/" + graph
            data = pd.read_csv("./results/" + filename + ".csv")
            #print("data", data)
            pyplot.boxplot(data)
            pyplot.gca().xaxis.set_ticklabels(data.columns)
            pyplot.title('boxplot of ' + graph + " on " + dataset)
            os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
            pyplot.savefig("./boxplot/" + filename + ".png")
            pyplot.show(block=False)
            pyplot.pause(1)
            pyplot.close('all')
