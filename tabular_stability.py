from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
import ape_tabular
import warnings
import time
from growingspheres.utils.gs_utils import distances, get_distances

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
    dataset_names = ["generate_moons", "compas", "titanic", "adult", "generate_moons", "generate_blob", "generate_blobs", "artificial", "blood", "diabete", "iris", "compas"]
    # array of the models used for the experiments
    models = [RandomForestClassifier(n_estimators=20, random_state=1),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('dt', tree.DecisionTreeClassifier())], voting="soft"),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                MLPClassifier(random_state=1)]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 50
    # Number of perturbed instances around the instances to explain for which we compute the test of unimodality
    number_of_perturb_instances = 5
    # The ratio of distance for the radius of the field
    ratio_radius = 5
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    verbose_ape = False
    # Threshold for explanation method precision
    threshold_interpretability = 0.99
    stability_name = ["Local Surrogate", "Anchors", 'APE']
    # Initialize all the variable needed to store the result in graph
    
    for dataset_name in dataset_names:
        if graph: 
            stability_informations = store_experimental_informations(len(models), len(stability_name), stability_name, len(models), columns_name_file3=['APE'])    
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, transformations = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            filename = "./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: 
                #experimental_informations.initialize_per_models(filename)
                stability_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
            dataset, black_box, x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name, model)
            print("###", model_name, "training on", dataset_name, "dataset.")
            
            black_box = black_box.fit(x_train, y_train)
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, black_box.predict, black_box_predict_proba=black_box.predict_proba,
                                                            continuous_features=continuous_features, 
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose_ape, threshold_precision=threshold_interpretability,
                                                            transformations=transformations)            
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                start_time = time.time()
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)

                #try:
                msi, csi, vsi = explainer.explain_instance(instance_to_explain, lime_stability=True, model_stability_index=True)
                print("msi, csi, vsi", msi, csi, vsi)
                if graph: stability_informations.store_experiments_information_instance(csi, 'csi_ape.csv', vsi, 'vsi_ape.csv', msi, 'msi_ape.csv' )
                """multimodal_result, original_features_employed = explainer.explain_instance(instance_to_explain, stability=True)
                farthest_distance = get_farthest_distance(instance_to_explain, x_train, categorical_features, metric='manhattan')
                if verbose:
                    print("distance la plus éloignée", farthest_distance)
                    print('features employed originaly', original_features_employed, "original multimodal", multimodal_result)
                if categorical_features != []:
                    perturb_instances, _ = generate_categoric_inside_ball(instance_to_explain, (0, farthest_distance/ratio_radius), (int) (100/ratio_radius), 
                                                                        number_of_perturb_instances, continuous_features, categorical_features, 
                                                                        categorical_values, feature_variance=explainer.feature_variance, 
                                                                        probability_categorical_feature=explainer.probability_categorical_feature, 
                                                                        libfolding=True, min_features=explainer.min_features,
                                                                        max_features=explainer.max_features)
                else:
                    perturb_instances = generate_inside_field(instance_to_explain, (0, farthest_distance/ratio_radius), 
                                                                        number_of_perturb_instances,
                                                                        feature_variance=explainer.feature_variance,
                                                                        max_features=explainer.max_features,
                                                                        min_features=explainer.min_features)

                # Test the stability error
                stability_results = 0
                nb_identical_features= 0
                for instance in perturb_instances:
                    multimodal, features_employed = explainer.explain_instance(instance, stability=True)
                    nb_identical_features += len(set(features_employed) & set(original_features_employed))
                    if multimodal_result == multimodal:
                        stability_results += 1
                    if verbose:
                        print("features employed by APE", features_employed, "multimodal result", multimodal)
                stability_results = stability_results/number_of_perturb_instances
                identical_features_score = nb_identical_features / (len(original_features_employed) * number_of_perturb_instances)
                if verbose:
                    print("proportion of identical features", identical_features_score)
                    print("stability score", stability_results)
                if graph: experimental_informations.store_stability_information_instance(stability_results, identical_features_score)
                """
                cnt += 1
                print("--- %s seconds to compute stability of APE for 1 instance ---" % (time.time() - start_time))
                #except Exception as inst:
                #    print(inst)

            filename_all = "./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: 
                #experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, filename_all=filename_all)
                stability_informations.store_experiments_information('csi_ape.csv', 'vsi_ape.csv', 
                                    'msi_ape.csv', filename_all=filename_all)
