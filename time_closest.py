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
#from keras.models import Sequential
#from keras.layers import Dense

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["blood"]#"diabetes"]#"generate_moons"]#"generate_circles"]#"generate_blob"]#"mega_generate_blobs"]#"generate_blobs"]#
    # array of the models used for the experiments
    models = [#RandomForestClassifier(n_estimators=20, random_state=1)]#, #LogisticRegression(),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1)]#,
                #RidgeClassifier(random_state=1)]#,
                #VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('rc', RidgeClassifier())], voting="hard")]#,
                #MLPClassifier(random_state=1)]
    #models = [RandomForestClassifier(n_estimators=20), MLPClassifier(random_state=1)]

    # Circles fait, Moons fait, Blob fait
    # Blobs fait, Blood fait, Diabetes fait
    # Mega Blobs fait

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 60
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
    interpretability_name = ['Growing Spheres', 'Growing Fields']
    #interpretability_name = ['ls log reg', 'ls raw data']
    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, \
            categorical_names, feature_names, transformations = generate_dataset(dataset_name)
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            if growing_sphere:
                filename = "./results/"+dataset_name+"/"+model_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (50% each set)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name)
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
                                                            feature_names=feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability, 
                                                            transformations=transformations)
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
                
                time_gs = explainer.explain_instance(instance_to_explain, time_k_closest=True,
                                                            growing_method="GS")
                time_gf = explainer.explain_instance(instance_to_explain, time_k_closest=True,
                                                            growing_method="GF")

                print("average time for GF", time_gf)
                print("average time for GS", time_gs)
                print("model", model_name)
                print("dataset", dataset_name)
                try:
                    print("GF", "better" if time_gf < time_gs else "worse", "than GS")
                    if graph: #experimental_informations.store_average_distance_instance(average_distance, average_distance_spheres)
                        experimental_informations.store_experiments_information_instance([time_gs, time_gf], 'average_time.csv')
                    cnt += 1
                except Exception as inst:
                    print(inst)
            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, 'average_time.csv',
                                                                        filename_all=filename_all)