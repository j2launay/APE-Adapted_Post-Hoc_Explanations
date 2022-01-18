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
from anchors import limes
from sklearn.metrics import precision_score
from growingspheres.utils.gs_utils import distances
from growingspheres import counterfactuals as cf

def find_closest_counterfactual(instance, explainer, method='GF', radius=False):
    # Computes the distance to the farthest instance from the training dataset to bound generating instances 
    farthest_distance = 0
    for training_instance in explainer.train_data:
        # get_distances is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
        # but it adds a sparsity distance computation (i.e: number of same values) 
        #farthest_distance_now = get_distances(training_instance, instance, categorical_features=self.categorical_features)["euclidean"]
        farthest_distance_now = distances(training_instance, instance, explainer)
        if farthest_distance_now > farthest_distance:
            farthest_distance = farthest_distance_now
    
    growing_sphere = cf.CounterfactualExplanation(instance, explainer.black_box_predict, method=method, target_class=None, 
                continuous_features=explainer.continuous_features, categorical_features=explainer.categorical_features, 
                categorical_values=explainer.categorical_values, max_features=explainer.max_features,
                min_features=explainer.min_features)
    growing_sphere.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, 
                verbose=explainer.verbose, feature_variance=explainer.feature_variance, 
                farthest_distance_training_dataset=farthest_distance, 
                probability_categorical_feature=explainer.probability_categorical_feature, 
                min_counterfactual_in_sphere=explainer.nb_min_instance_per_class_in_sphere)
    if radius == False:
        return growing_sphere.enemy # Return the closest counterfactual
    else:
        return growing_sphere.enemy, growing_sphere.radius

def get_farthest_distance(instance, train_data, categorical_features, explainer, metric='euclidean'):
    print("TEST")
    farthest_distance = 0
    for training_instance in train_data:
        # get_distance is similar to pairwise distance (i.e: it is the same results for euclidean distance) 
        # but it adds a sparsity distance computation (i.e: number of same values)
        if 'manhattan' in metric:
            farthest_distance_now = distances(training_instance, instance, explainer)
        else:
            farthest_distance_now = get_distances(training_instance, instance, categorical_features=categorical_features)[metric]
        if farthest_distance_now > farthest_distance:
            farthest_distance = farthest_distance_now
    return farthest_distance

def compute_precision_in_sphere(ape_tabular, radius_sphere, closest_counterfactual, farthest_distance, target_class, 
                                lime_explainer, linear_explainer='ls'):
    position_instances_in_sphere, nb_training_instance_in_sphere = ape_tabular.instances_from_dataset_inside_sphere(closest_counterfactual, 
                                                                                                radius_sphere, ape_tabular.test_data)
    ape_tabular.target_class = target_class
    try:
        instances_in_sphere, labels_in_sphere, percentage_distribution, _ = \
                            ape_tabular.generate_instances_inside_sphere(radius_sphere, 
                                                                    closest_counterfactual, 
                                                                    dataset=ape_tabular.test_data,
                                                                    farthest_distance=farthest_distance, 
                                                                    min_instance_per_class=ape_tabular.nb_min_instance_per_class_in_sphere,
                                                                    position_instances_in_sphere=position_instances_in_sphere, 
                                                                    nb_training_instance_in_sphere=nb_training_instance_in_sphere,
                                                                    lime_ls=True)
    except:
        return 0
    ape_tabular.nb_min_instance_in_sphere = 800
    if linear_explainer == 'ls':
        linear_model = lime_explainer.explain_instance_training_dataset(closest_counterfactual, ape_tabular.black_box_predict, 
                                                                    num_features=6, model_regressor = LogisticRegression(),
                                                                    ape=ape_tabular, instances_in_sphere=instances_in_sphere)
    elif linear_explainer =='lime':
        linear_model = lime_explainer.explain_instance(instance_to_explain, ape_tabular.black_box_predict_proba, num_features=6)#, 
                                                                    #model_regressor = LogisticRegression())
    else:
        linear_model = lime_explainer.explain_instance(closest_counterfactual, ape_tabular.black_box_predict, 
                                                                    model_regressor=LogisticRegression(), 
                                                                    num_features=6)
    prediction_inside_sphere = ape_tabular.modify_instance_for_linear_model(linear_model, instances_in_sphere)
    #precision_local_surrogate = ape_tabular.compute_linear_regression_precision(prediction_inside_sphere, labels_in_sphere)
    #print(prediction_inside_sphere)
    if linear_explainer == 'lime':
        new_prediction_inside_sphere = []
        for prediction in prediction_inside_sphere:
            if prediction > 0.5:
                new_prediction_inside_sphere.append(ape_tabular.target_class[0])
            else:
                new_prediction_inside_sphere.append(1-ape_tabular.target_class[0])
        prediction_inside_sphere = new_prediction_inside_sphere
        #print("new prediction", prediction_inside_sphere)

    precision_local_surrogate = max(precision_score(labels_in_sphere, prediction_inside_sphere, pos_label=ape_tabular.target_class[0]),\
        precision_score(labels_in_sphere, prediction_inside_sphere, pos_label=1-ape_tabular.target_class[0]))
    return precision_local_surrogate

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_moons", "generate_circles", "blood", "diabete", "generate_blobs"]#, "iris", "artificial"]
    # array of the models used for the experiments
    models = [GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                RandomForestClassifier(n_estimators=20,random_state=1), 
                #MLPClassifier(random_state=1, activation="logistic"),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('mlp', MLPClassifier())], voting="soft"),
                MLPClassifier(random_state=1)]#,
                #RidgeClassifier()]
    #models = [RandomForestClassifier(n_estimators=20), RidgeClassifier()]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 5
    # Print explanation result
    illustrative_example = False
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    # Threshold for explanation method precision
    threshold_interpretability = 0.99
    linear_models_name = ['local surrogate', 'lime extending', 'lime regression', 'lime not binarize', 'lime traditional']
    interpretability_name = ['radius', 'Lime', 'Local Surrogate']
    #interpretability_name = ['ls log reg', 'ls raw data']
    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, transformations = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            filename = "./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
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
            black_box_labels = predict(x_train)
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, predict, black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability, 
                                                            transformations=transformations)

            lime_explainer = limes.lime_tabular.LimeTabularExplainer(x_train, feature_names=dataset.feature_names, 
                                                                categorical_features=categorical_features, categorical_names=categorical_names,
                                                                class_names=class_names, discretize_continuous=True, discretizer='MDLP', 
                                                                training_labels=black_box_labels)
            
            local_surrogate_best_ratio = 0
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
                try:
                    test += 2
                except:
                    closest_counterfactual = find_closest_counterfactual(instance_to_explain, explainer)
                    target_class = predict(instance_to_explain.reshape(1, -1))
                    opponent_class = predict(closest_counterfactual.reshape(1, -1))
                    farthest_distance = get_farthest_distance(instance_to_explain, x_train, categorical_features, explainer, metric='manhattan')
                    
                    for radius in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1):
                        explainer.nb_min_instance_in_sphere = 800
                        local_surrogate_precision = compute_precision_in_sphere(explainer, radius, closest_counterfactual, farthest_distance, 
                                                        target_class, lime_explainer)
                        explainer.nb_min_instance_in_sphere = 800
                        lime_precision = compute_precision_in_sphere(explainer, radius, instance_to_explain, farthest_distance,
                                                        target_class, lime_explainer, linear_explainer='lime')
                        print("ls precision", local_surrogate_precision)
                        print("lime precision", lime_precision)
                        print("radius", radius)
                        if local_surrogate_precision >= lime_precision:
                            #print("YEAH")
                            local_surrogate_best_ratio += 1
                        if graph: 
                            experimental_informations.store_experiments_information_instance([radius, lime_precision, 
                                        local_surrogate_precision], 'lime_vs_ls_radius.csv')
                    cnt += 1
                #except Exception as inst:
                #    print(inst)
            filename_all = "./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            
            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, 
                        nb_model, 'lime_vs_ls_radius.csv', filename_all=filename_all)
