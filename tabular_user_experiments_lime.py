from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations, prepare_legends
import baseGraph
import ape_tabular
import warnings
import random
from ape_tabular_experiments import simulate_user_experiments_lime_ls, modify_dataset, decision_tree_function
from sklearn import tree

def compute_score_interpretability_method(features_employed_by_explainer, features_employed_black_box):
    """
    Compute the score of the explanation method based on the features employed for the explanation compared to the features truely used by the black box
    """
    score = 0
    for feature_employe in features_employed_by_explainer:
        if feature_employe in features_employed_black_box:
            score += 1
    return score/len(features_employed_by_explainer)

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Dataset used for the experiments
    dataset_names = ["compas", "generate_blobs", "adult", "diabete"]
    # Black box models for which we generate explanation
    models = [tree.DecisionTreeClassifier(max_depth=4), tree.DecisionTreeClassifier(), LogisticRegression()]
    models_name = ['DecisionTreeClassifier_depth4', 'DecisionTreeClassifier', 'LogisticRegression']
    models = [LogisticRegression(), tree.DecisionTreeClassifier()]
    models_name = ['LogisticRegression', 'DecisionTreeClassifier']
    # Number of feature from the dataset that are modified (values are set to 0 to train the decision model)
    nb_feature_to_modify = 6
    # If set to True store the results inside a graph
    graph = True
    # Number of instance for which explanations are computed
    max_instance_to_explain = 50
    # If set to True print detailed information
    verbose = False
    # Precision threshold for explanation models and linear separability test 
    threshold_interpretability = 0.85
    # Name of the explanation method printed on the graphs
    interpretability_name = ['lime', 'random', 'Local Surrogate']

    # Initialize variable to store the results for the graph representation
    if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
    for dataset_name in dataset_names:
        # Store dataset information such as class names and the list of categerical features as well as variables (x for input and y for labels)
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            cnt = 0
            model_name = models_name[nb_model]
            filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            # Split the dataset in test and train set (50% each)
            dataset, black_box, x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name, model)
            print("###", model_name, "training on", dataset_name, "dataset.")
            # Modify the dataset to train the "black box" model only on a subset of features
            x_train_bb, feature_kept = modify_dataset(x_train, nb_feature_to_modify)

            black_box = black_box.fit(x_train_bb, y_train)
            if "Tree" in model_name and verbose:
                print("features importances", black_box.feature_importances_)
            elif verbose:
                print("features importances", black_box.coef_)
            
            print('### Accuracy:', sum(black_box.predict(x_test) == y_test)/len(y_test))
            explainer = ape_tabular.ApeTabularExplainer(x_test, class_names, black_box.predict, black_box.predict_proba, continuous_features=continuous_features, 
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, linear_separability_index=threshold_interpretability)
            for instance_to_explain in x_test: 
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt, "over", max_instance_to_explain)
                print("### Models ", nb_model+1, "over", len(models))
                
                if "Tree" in model_name:
                    features_employed_black_box = decision_tree_function(model, instance_to_explain.reshape(1, -1))
                else:
                    features_employed_black_box = feature_kept
                if verbose: print("features employed by the black box", features_employed_black_box)
                print("features employed by the black box", features_employed_black_box)

                
                try:
                    test += 2
                except:
                    # Get the list of features employed by Lime and Local Surrogate 
                    features_employed_in_lime, features_employed_in_local_surrogate = explainer.explain_instance(instance_to_explain, 
                                                                lime_vs_local_surrogate=True, nb_features_employed=len(features_employed_black_box))
                    if features_employed_in_lime == []:
                        continue

                    # Selects randomly as many features as the black box model is actually chosen among all the features
                    random_explainer = random.sample(range(len(instance_to_explain)), len(features_employed_black_box))
                    if verbose:
                        print("features employed by Local Surrogate", features_employed_in_local_surrogate)
                        print("features employed by lime", features_employed_in_lime)
                        print("features employed randomly", random_explainer)
                    print("features employed by Local Surrogate", features_employed_in_local_surrogate)
                    print("features employed by lime", features_employed_in_lime)
                    print("features employed randomly", random_explainer)

                    score_lime = compute_score_interpretability_method(features_employed_in_lime, features_employed_black_box)
                    score_local_surrogate = compute_score_interpretability_method(features_employed_in_local_surrogate, features_employed_black_box)
                    score_random = compute_score_interpretability_method(random_explainer, features_employed_black_box)
                    cnt += 1

                #except Exception as inst:
                #    print(inst)

                if graph: experimental_informations.store_user_experiments_information_instance([score_lime, score_random, score_local_surrogate])
            filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.store_user_experiments_information(max_instance_to_explain, nb_model, filename_all=filename_all, lime=True)

            if graph:
                plt.show(block=False)
                plt.pause(1)
                plt.close('all')
                graph_score = baseGraph.BaseGraph(title="Results of simulated user experiments for Lime, Local Surrogate and a baseline.", y_label="Recall score", 
                                        model=model_name, accuracy=black_box.score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
                color = ['black', 'red', 'green', 'blue', 'cyan', 'yellow']
                graph_score.show_coverage(model=interpretability_name, mean_coverage=experimental_informations.final_recall, color=color[:len(interpretability_name)], title="Recall lime ls")

        if len(models) > 1 and graph:
            # In case of multiple model used to classify we store in graph all the results in order to compare the impact of the black box model
            color, bars, y_pos = prepare_legends(experimental_informations.final_recalls, models, interpretability_name)
            plt.show(block=False)
            plt.pause(1)
            plt.close('all')
            graph_models_coverage = baseGraph.BaseGraph(title="Results of simulated user experiments for Lime, Local Surrogate and a baseline on multiple models",
                                        y_label="Recall", model=model_name, accuracy=black_box.score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
            
            graph_models_coverage.show_multiple_models(models_name=models_name, interpretability_name=interpretability_name, 
                                        mean=experimental_informations.final_recalls, color=color, 
                                        title="recall lime ls", bars=bars, y_pos=y_pos)
