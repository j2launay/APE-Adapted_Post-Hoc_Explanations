from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from anchors import limes
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations, prepare_legends
from lime_vs_local_surrogate import find_closest_counterfactual, compute_precision_in_sphere, get_farthest_distance
import baseGraph
import ape_tabular
import warnings
import pickle

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = [ "compas", "titanic", "adult", "blood", "diabete", "generate_moons", "generate_blob", "generate_blobs",]
    # array of the models used for the experiments
    models = [RandomForestClassifier(n_estimators=20), #LogisticRegression(),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0),
                #tree.DecisionTreeClassifier(),
                RidgeClassifier(),
                #Sequential(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('rc', RidgeClassifier())], voting="hard"),
                MLPClassifier(random_state=1)]
    #models=[RidgeClassifier(), MLPClassifier(random_state=1)]
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
        label_graph = "growing spheres "
        growing_method = "GS"
    else:
        label_graph = ""
        growing_method = "GF"
    # Threshold for explanation method precision
    threshold_interpretability = 0.99
    linear_separability_index = 0.99
    linear_models_name = ['local surrogate', 'lime extending', 'lime regression', 'lime not binarize', 'lime traditional']
    interpretability_name = ['Local Surrogate']
    #interpretability_name = ['ls log reg', 'ls raw data']
    # Initialize all the variable needed to store the result in graph
    if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
    for dataset_name in dataset_names:
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            if growing_sphere:
                filename = "./results/"+dataset_name+"/"+model_name+"/growing_spheres/"+str(threshold_interpretability)+"/ls_"
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/ls_"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/ls_"
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/ls_"
            if graph: experimental_informations.initialize_per_models(filename)
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
            
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, predict, #black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability,
                                                            linear_separability_index=linear_separability_index)
            
            linear_explainer = limes.lime_tabular.LimeTabularExplainer(x_train, feature_names=dataset.feature_names, 
                                                                categorical_features=categorical_features, categorical_names=categorical_names,
                                                                class_names=class_names, discretize_continuous=True, discretizer="MDLP", 
                                                                training_labels=predict(x_train))
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
                closest_counterfactual, radius = find_closest_counterfactual(instance_to_explain, explainer, method='GS')
                target_class = predict(instance_to_explain.reshape(1, -1))
                opponent_class = predict(closest_counterfactual.reshape(1, -1))
                farthest_distance = get_farthest_distance(instance_to_explain, x_train, categorical_features, explainer, metric='manhattan')
                
                explainer.nb_min_instance_in_sphere = 800
                local_surrogate_precision = compute_precision_in_sphere(explainer, radius, closest_counterfactual, farthest_distance, 
                                                    target_class, linear_explainer, linear_explainer='local')
                #local_surrogate_coverage = 
                print("ls precision", local_surrogate_precision)
                #try:
                #precision, coverage, f2, multimodal_result = linear_explainer.explain_instance(instance_to_explain, growing_method=growing_method, 
                #                                            all_explanations_model=True)
                #print("precision", precision)
                #print("coverage", coverage)
                #print("f2", f2)
                if graph: experimental_informations.store_experiments_information_instance(local_surrogate_precision, [], [], local_surrogate=True)
                cnt += 1
                #except Exception as inst:
                #    print(inst)

            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, 
                            filename_all=filename_all)#, local_surrogate=True)
            