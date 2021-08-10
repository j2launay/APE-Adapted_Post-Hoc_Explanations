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
from matplotlib.font_manager import FontProperties

thresholds_interpretability = ["0.99", "0.8"]
datasets = ["generate_moons", "compas", "titanic", "adult", "generate_moons", "generate_blob", "generate_blobs", "artificial", "blood", "diabete", "iris"]
models = [RandomForestClassifier(), LogisticRegression(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('dt', tree.DecisionTreeClassifier())], voting="soft"),
                Sequential(),
                GradientBoostingClassifier(),
                tree.DecisionTreeClassifier(), 
                RidgeClassifier(), 
                MLPClassifier(),
                None]
graphs = ["coverage", "precision", "f1", 'recall', "average_distance", "coverages", "precisions", "f1s", "distance", 
            'recalls_lime', 'recall_lime', "degrees", "counterfactual_in_anchor", "kendall", "mean_top_k"]
graphs_lime = ["lime_vs_ls", "lime_ls"]

for threshold_interpretability in thresholds_interpretability:
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
                    print("filename", filename)
                    data = pd.read_csv("./results/" + filename + ".csv")
                    radius = data["radius"].unique()
                    data = data.drop(['radius'], axis=1)
                    pyplot.xlabel('radius')
                    pyplot.ylabel('precision')
                    pyplot.ylim(0, 1.1)
                    pyplot.title('boxplot of ' + graph + " on " + dataset + " for " + model[:-1])
                    first = True
                    for nb, rad in enumerate(radius):
                        values = data.iloc[nb::radius.shape[0]]
                        values_lime = values["Lime"].to_numpy()
                        values_ls = values["Local Surrogate"].to_numpy()
                        if first:
                            pyplot.errorbar(rad + 0.005, np.mean(values_lime), yerr=np.var(values_lime), color='b', marker='X', label = 'Lime')
                            pyplot.errorbar(rad - 0.005, np.mean(values_ls), yerr=np.var(values_ls), color='r', marker='o', label = 'Local Surrogate')
                            first = False
                        else:
                            pyplot.errorbar(rad + 0.005, np.mean(values_lime), yerr=np.var(values_lime), color='b', marker='X')
                            pyplot.errorbar(rad - 0.005, np.mean(values_ls), yerr=np.var(values_ls), color='r', marker='o')
                    fontP = FontProperties()
                    pyplot.legend(bbox_to_anchor=(0.35, 1.6), loc='upper left', prop=fontP)
                    os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
                    pyplot.savefig("./boxplot/" + filename + ".png")
                    pyplot.show(block=False)
                    pyplot.pause(0.5)
                    pyplot.close('all')
                except Exception as inst:
                    print(inst)


for threshold_interpretability in thresholds_interpretability:
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
                    pyplot.title('boxplot of ' + graph + " on " + dataset + " for " + model[:-1])
                    os.makedirs(os.path.dirname("./boxplot/" + filename), exist_ok=True)
                    pyplot.savefig("./boxplot/" + filename + ".png")
                    pyplot.show(block=False)
                    pyplot.pause(0.5)
                    pyplot.close('all')
                except Exception as inst:
                    print(inst)
