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

threshold_interpretability = "0.99"
datasets = ["generate_moons"]#, "compas", "titanic", "adult", "generate_moons", "generate_blob", "generate_blobs", "artificial", "blood", "diabete", "iris"]
models = [RandomForestClassifier(), LogisticRegression(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('dt', tree.DecisionTreeClassifier())], voting="soft"),
                Sequential(),
                GradientBoostingClassifier(),
                tree.DecisionTreeClassifier(), 
                RidgeClassifier(), 
                MLPClassifier(),
                None]
graphs = ["coverage", "precision", "f1", 'recall', "average_distance", "coverages", "precisions", "f1s", "distance"]
graphs_lime = ["lime_vs_ls", "lime_ls"]

for dataset in datasets:
    print(dataset)
    for model in models:
        if model is None:
            model = ""
        else:
            model = type(model).__name__ + "/"
        print(model)
        for graph in graphs_lime:
            try:
                pyplot.subplot(212)
                filename = dataset + "/" + model + threshold_interpretability + "/" + graph
                data = pd.read_csv("./results/" + filename + ".csv")
                radius = data["radius"].unique()
                data = data.drop(['radius'], axis=1)
                first = True
                pyplot.xlabel('radius of the fields')
                pyplot.ylabel('precision of the linear explanation')
                pyplot.ylim(0, 1.1)
                pyplot.title('boxplot of ' + graph + " on " + dataset + " for " + model)
                while not data.empty:
                    data_split = data.head(len(radius))
                    data = data.iloc[len(radius):, :]
                    if first:
                        pyplot.plot(radius, data_split["Lime"].to_numpy(), 'b-', label = 'Lime')
                        pyplot.plot(radius, data_split["Local Surrogate"].to_numpy(), 'r-', label = 'Local Surrogate')
                        first = False
                    else:
                        pyplot.plot(radius, data_split["Lime"].to_numpy(), 'b-')
                        pyplot.plot(radius, data_split["Local Surrogate"].to_numpy(), 'r-')

                pyplot.legend()
                os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
                pyplot.savefig("./boxplot/" + filename + ".png")
                pyplot.show(block=False)
                pyplot.pause(1)
                pyplot.close('all')
            except Exception as inst:
                print(inst)


for dataset in datasets:
    print(dataset)
    for model in models:
        if model is None:
            model = ""
        else:
            model = type(model).__name__ + "/"
        for graph in graphs:
            try:
                pyplot.subplot(212)
                filename = dataset + "/" + model + threshold_interpretability + "/" + graph
                data = pd.read_csv("./results/" + filename + ".csv")
                #print("data", data)
                pyplot.boxplot(data)
                pyplot.ylim(0, 1.1)
                pyplot.gca().xaxis.set_ticklabels(data.columns)
                pyplot.title('boxplot of ' + graph + " on " + dataset + " for " + model)
                os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
                pyplot.savefig("./boxplot/" + filename + ".png")
                pyplot.show(block=False)
                pyplot.pause(1)
                pyplot.close('all')
            except Exception as inst:
                print(inst)
