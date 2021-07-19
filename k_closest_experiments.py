from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations, prepare_legends
import baseGraph
import ape_tabular
import warnings
import pickle
from keras.models import Sequential
from keras.layers import Dense

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_blob", "generate_moons", "generate_blobs", "artificial", "compas", "titanic", "adult", "blood", "diabete", "iris"]
    # array of the models used for the experiments
    models = [RandomForestClassifier(n_estimators=20), LogisticRegression(),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0),
                tree.DecisionTreeClassifier(), 
                RidgeClassifier(), 
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('rc', RidgeClassifier())], voting="hard"),
                MLPClassifier(random_state=1), 
                Sequential()]
    models = [RandomForestClassifier(n_estimators=20), MLPClassifier(random_state=1), Sequential()]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 5
    # Print explanation result
    illustrative_example = False
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    growing_sphere = False
    if growing_sphere:
        label_graph = "growing_spheres"
        growing_method = "GS"
    else:
        label_graph = ""
        growing_method = "GF"
    # Threshold for explanation method precision
    threshold_interpretability = 0.99
    linear_models_name = ['local surrogate', 'lime extending', 'lime regression', 'lime not binarize', 'lime traditional']
    interpretability_name = ['Growing Fields', 'Growing Spheres']
    #interpretability_name = ['ls log reg', 'ls raw data']
    # Initialize all the variable needed to store the result in graph
    if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name)
    for dataset_name in dataset_names:
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            if graph: experimental_informations.initialize_per_models()
            model_name = type(model).__name__
            models_name.append(model_name)
            # Split the dataset inside train and test set (50% each set)
            dataset, black_box, x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name, model)
            print("###", model_name, "training on", dataset_name, "dataset.")
            if 'Sequential' in model_name:
                # Train a neural network classifier with 2 relu and a sigmoid activation function
                black_box.add(Dense(12, input_dim=len(x_train[0]), activation='relu'))
                black_box.add(Dense(8, activation='relu'))
                black_box.add(Dense(1, activation='sigmoid'))
                black_box.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                black_box.fit(x_train, y_train, epochs=50, batch_size=10)
                def predict(x):
                    if x.shape[0] > 1:
                        return np.asarray([prediction[0] for prediction in black_box.predict_classes(x)])
                    return black_box.predict_classes(x)[0]
                def score(x, y):
                    return sum(predict(x) == y)/len(y)
            else:
                black_box = black_box.fit(x_train, y_train)
                predict = black_box.predict
                score = black_box.score
            print('### Accuracy:', score(x_test, y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, predict, black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability)
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
                
                average_distance, all_average_distance, average_distance_spheres, all_average_distance_spheres = explainer.explain_instance(instance_to_explain, 
                                                            growing_method=growing_method, k_closest=True)
                print("average distance for GF", average_distance)
                print("average distance for GS", average_distance_spheres)
                if average_distance != average_distance_spheres:
                    print("GF", "better" if average_distance < average_distance_spheres else "worse", "than GS")
                if graph: experimental_informations.store_average_distance_instance(average_distance, average_distance_spheres)
                cnt += 1
            if growing_sphere:
                filename="./results/"+dataset_name+"/"+model_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, filename=filename)
            """
            if graph:
                plt.show(block=False)
                plt.pause(1)
                plt.close('all')
                color = ['black', 'red', 'green', 'blue', 'cyan', 'yellow']
                
                graph_coverage = baseGraph.BaseGraph(title="Results of coverage for LS, APE and Anchors ", y_label="Coverage", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
                graph_coverage.show_coverage(model=interpretability_name, mean_coverage=experimental_informations.final_coverage, 
                                        color=color[:len(interpretability_name)], title= label_graph + "coverage")
                
                graph_roc = baseGraph.BaseGraph(title="Results of accuracy score for LS, APE and Anchors", y_label="Precision", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
                graph_roc.show_coverage(model=interpretability_name, mean_coverage=experimental_informations.final_precision, 
                                        color=color[:len(interpretability_name)], title= label_graph + "Precision")
                
                graph_f1 = baseGraph.BaseGraph(title="Results of F1 score for LS, APE and Anchors", y_label="F1 score", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
                graph_f1.show_coverage(model=interpretability_name, mean_coverage=experimental_informations.final_f1, 
                                        color=color[:len(interpretability_name)], title= label_graph + "f1")
                
        if len(models) > 1 and graph:
            # In case of multiple models we compare results for each model
            color, bars, y_pos = prepare_legends(experimental_informations.final_coverages, models, interpretability_name)
            plt.show(block=False)
            plt.pause(1)
            plt.close('all')
            graph_models_coverage = baseGraph.BaseGraph(title="Results of coverage for LS, APE and Anchors on multiple models", y_label="Coverage", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
            graph_models_coverage.show_multiple_models(models_name=models_name, interpretability_name=interpretability_name, 
                                        mean=experimental_informations.final_coverages, color=color, 
                                        title= label_graph + "coverage", bars=bars, y_pos=y_pos)

            graph_models_precision = baseGraph.BaseGraph(title="Results of precision for LS, APE and Anchors on multiple models", y_label="Precision", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
            graph_models_precision.show_multiple_models(models_name=models_name, interpretability_name=interpretability_name, 
                                        mean=experimental_informations.final_precisions, color=color, 
                                        title= label_graph + "precision", bars=bars, y_pos=y_pos)
            
            graph_models_f1 = baseGraph.BaseGraph(title="Results of F1 score for LS, APE and Anchors on multiple models", y_label="F1", 
                                        model=model_name, accuracy=score(x_test, y_test), 
                                        dataset=dataset_name, threshold=threshold_interpretability)
            graph_models_f1.show_multiple_models(models_name=models_name, interpretability_name=interpretability_name, 
                                        mean=experimental_informations.final_f1s, color=color, 
                                        title= label_graph + "F1 score", bars=bars, y_pos=y_pos)

            y_pos = range(len(interpretability_name))
            graph_models_multimodal = baseGraph.BaseGraph(title="Proportion of times APE returns a multimodal explanation over multiple models", y_label="Multimodal",
                                        model=model_name, accuracy=score(x_test, y_test),
                                        dataset=dataset_name, threshold=threshold_interpretability)
            graph_models_multimodal.show_proportion_multimodal(model=models_name, 
                                        proportions_multimodal=experimental_informations.final_multimodals, 
                                        color=color[:len(models)], title= label_graph + "Proportion of Multimodal")
            """