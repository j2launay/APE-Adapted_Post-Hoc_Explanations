from growingspheres import counterfactuals as cf
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
from growingspheres.utils.gs_utils import distances
import scipy.stats as stats

def get_farthest_distance(instance, train_data, categorical_features, metric='euclidean'):
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

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_blob", "generate_moons", "generate_blobs", "titanic", "adult", "blood", "diabete", "iris", "artificial", "compas"]
    # array of the models used for the experiments
    models = [RandomForestClassifier(n_estimators=20), LogisticRegression(),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0),
                tree.DecisionTreeClassifier(),
                #Sequential(),
                MLPClassifier(random_state=1)]
    #models = [LogisticRegression(), RandomForestClassifier(n_estimators=20)]#, Sequential()]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 50
    threshold_interpretability = 0.99
    nb_feature_linear_explanation = 5
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
    interpretability_name = ['direction vector', 'linear explanation', "degree"]
    interpretability_name_kendall = ['top k direction vector', 'top k linear explanation', "kendall", "hit-1", "hit-2", "hit-3", "hit-4", "hit-5"]
    anchor_name = ['anchor rule', 'closest counterfactual', 'presence']
    # Initialize all the variable needed to store the result in graph
    if graph: 
        experimental_informations_degrees = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        experimental_informations_kendall = store_experimental_informations(len(models), len(interpretability_name_kendall), 
                                                                                    interpretability_name_kendall, len(models))
        experimental_informations_anchor = store_experimental_informations(len(models), len(anchor_name), 
                                                                                    anchor_name, len(models))    
    for dataset_name in dataset_names:
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            if growing_sphere:
                filename = "./results/"+dataset_name+"/"+model_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: 
                experimental_informations_degrees.initialize_per_models(filename)
                experimental_informations_kendall.initialize_per_models(filename)
                experimental_informations_anchor.initialize_per_models(filename)
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
            mean_top_k = np.zeros(nb_feature_linear_explanation)
            nb_top_k = np.zeros(nb_feature_linear_explanation)
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                #print("instance to explain:", instance_to_explain)
                try:
                    farthest_distance = get_farthest_distance(instance_to_explain, x_train, categorical_features, metric='manhattan')
                    #print("farthest distance", farthest_distance)
                    growing_fields = cf.CounterfactualExplanation(instance_to_explain, predict, method=growing_method, target_class=None, 
                                        continuous_features=continuous_features, categorical_features=categorical_features, 
                                        categorical_values=categorical_values, min_features=explainer.min_features,
                                        max_features=explainer.max_features)
                    growing_fields.fit(verbose=verbose, feature_variance=explainer.feature_variance, 
                                        farthest_distance_training_dataset=farthest_distance, 
                                        probability_categorical_feature=explainer.probability_categorical_feature, 
                                        min_counterfactual_in_sphere=explainer.nb_min_instance_per_class_in_sphere)
                    closest_counterfactual = growing_fields.enemy
                    #print("closest counterfactual", closest_counterfactual)
                    direction_vector = instance_to_explain - closest_counterfactual
                    #print("closest counterfactual", closest_counterfactual)
                    #print('direction vector', direction_vector)
                    explainer.target_class = model.predict(instance_to_explain.reshape(1, -1))[0]
                    position_instances_in_sphere, nb_training_instance_in_sphere = explainer.instances_from_dataset_inside_sphere(growing_fields.enemy, 
                                                                                                                    growing_fields.radius, x_train)
                    #print("position in sphere", position_instances_in_sphere[:10])
                    instances_in_sphere, _, _, _ = explainer.generate_instances_inside_sphere(growing_fields.radius, 
                                                        growing_fields.enemy,  x_test, farthest_distance, 
                                                        explainer.nb_min_instance_per_class_in_sphere,
                                                        position_instances_in_sphere, nb_training_instance_in_sphere)
                    #print("instances in sphere", instances_in_sphere)                
                    ls_raw_data = explainer.lime_explainer.explain_instance_training_dataset(closest_counterfactual, black_box.predict_proba, 
                                                                    num_features=nb_feature_linear_explanation, instances_in_sphere = instances_in_sphere)
                    #print("interpretability method")
                    anchors = explainer.anchor_explainer.explain_instance(instance_to_explain, explainer.black_box_predict, 
                                        threshold=explainer.threshold_precision, 
                                        delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, 
                                        stop_on_first=False, desired_label=None, beam_size=4)
                    target_prediction = np.array(black_box.predict_proba(instance_to_explain.reshape(1, -1)))
                    target_class = np.argmax(target_prediction)
                    new_dataset = np.insert(x_test, 0, np.array(closest_counterfactual), 0)
                    rules, pandas_frame = explainer.generate_rule_and_data_for_anchors(anchors.names(), target_class, new_dataset)
                    instances_in_anchors = explainer.get_base_model_data(rules, pandas_frame)
                    test = (instances_in_anchors == np.array(closest_counterfactual)).all(1).any()
                    if not test:
                        #print("YEAH C'EST SIMPLE LE COUNTERFACTUAL N'EST PAS DANS L'ANCHOR")
                        counterfactual_in_anchors = 1
                    else:
                        #print("BOUH :'(")
                        counterfactual_in_anchors = 0

                    impact_vector = np.zeros(len(closest_counterfactual))
                    for nb, explanation in enumerate(ls_raw_data.as_list()):
                        for nb_feature, element in enumerate(dataset.feature_names):
                            split_x = explanation[0].split(element)
                            if len(split_x) > 1:
                                impact_vector[nb_feature] = explanation[1]
                    
                    impact_vector = -np.asarray(impact_vector)
                    #print("impact vector", impact_vector)
                    nb_feature_to_compare = min(len(np.where([x != 0 for x in direction_vector])[0]), len(np.where([x > 0  for x in impact_vector])[0]))
                    top_k_impact_vector = impact_vector.argsort()[nb_feature_to_compare:][::-1]
                    top_k_direction_vector = direction_vector.argsort()[nb_feature_to_compare:][::-1]
                    #print("top k impact vector", top_k_impact_vector)
                    #print("top k direction vector", top_k_direction_vector)
                    sum_same = 0
                    hit_k = [-1]*nb_feature_linear_explanation
                    for nb, feature in enumerate(top_k_impact_vector[:5]):
                        if feature in top_k_direction_vector[:nb]:
                            #print("TEST", feature, "from", top_k_impact_vector[:nb])
                            #print("in", top_k_direction_vector)
                            #print()
                            sum_same += 1
                        mean_top_k[nb] = mean_top_k[nb] + sum_same
                        nb_top_k[nb] += 1
                        hit_k[nb] = sum_same
                    
                    #print("explication de LIME", ls_raw_data.as_list())
                    unit_vector_1 = direction_vector / np.linalg.norm(direction_vector)
                    unit_vector_2 = impact_vector / np.linalg.norm(impact_vector)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle = np.arccos(dot_product)
                    #print("angle", np.degrees(angle))
                    tau, p_value = stats.kendalltau(top_k_impact_vector, top_k_direction_vector)
                    if graph: 
                        experimental_informations_degrees.store_degrees([direction_vector, impact_vector, angle])
                        experimental_informations_kendall.store_kendall([top_k_direction_vector, top_k_impact_vector, tau] + hit_k)
                        experimental_informations_anchor.store_counterfactual_in_anchor([anchors.names(), closest_counterfactual, counterfactual_in_anchors])
                        #print("top k mean", mean_top_k)
                        #print("nb top k", nb_top_k)
                        #print(mean_top_k / nb_top_k)
                        if graph: experimental_informations_kendall.store_mean_top_k(mean_top_k/nb_top_k)
                    cnt += 1
                except Exception as inst:
                    print(inst)

            if graph: 
                experimental_informations_degrees.store_experiments_information(max_instance_to_explain, nb_model, filename_all=filename_all)
                experimental_informations_kendall.store_experiments_information(max_instance_to_explain, nb_model, filename_all=filename_all)
                experimental_informations_anchor.store_experiments_information(max_instance_to_explain, nb_model, filename_all=filename_all)
