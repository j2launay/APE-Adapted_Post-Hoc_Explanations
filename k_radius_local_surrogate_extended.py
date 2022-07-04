from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
import ape_tabular
import warnings

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_moons"]#"mega_generate_blobs"]#"diabetes"]#"generate_blob"]#"generate_circles"]#"generate_moons"]#"generate_blobs"]#"cancer"]#
    # blob, circles, moons, blobs, cancer, diabetes, blood, mega blobs 
    # array of the models used for the experiments
    models = [GaussianNB()]#,
                #RandomForestClassifier(n_estimators=20, random_state=1), 
                #GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1)]#,
                #VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB())], voting="soft"),
                #MLPClassifier(random_state=1, activation='logistic')]
    # Mega Blobs, Blood, Blob, Moons, Circles, Cancer, Diabetes, blobs
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 2
    k_radius_local_surrogate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    """ All the variable necessaries for generating the graph results """
    # Store results inside csv files if set to True
    graph = True
    verbose = False
    growing_sphere = False
    growing_method = "GS" if growing_sphere else "GF"
    # Threshold for explanation method precision
    threshold_interpretability = 0.95
    linear_metric, temp_interpretability_name = ["", "cf", "auc", "auc cf"], k_radius_local_surrogate*4
    print(temp_interpretability_name)
    interpretability_name = []
    for compteur, x in enumerate(temp_interpretability_name):
        if compteur < 10:
            print(str(x) + " " + linear_metric[0])
            interpretability_name += [str(x) + " " + linear_metric[0]]
        elif compteur < 20: 
            interpretability_name += [str(x) + " " + linear_metric[1]]
        elif compteur < 30:
            interpretability_name += [str(x) + " " + linear_metric[2]]
        else: 
            interpretability_name += [str(x) + " " + linear_metric[3]]
    interpretability_name += ["radius", "fr pvalue", "cf pvalue", "separability", "fr fold",
                                    "cf fold", "bb", "dataset"]
    print(interpretability_name)
    # Initialize all the variable needed to store the result in graph

    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(interpretability_name)
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
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y)
            print("###", model_name, "training on", dataset_name, "dataset.")
            black_box = black_box.fit(x_train, y_train)
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, black_box.predict, black_box.predict_proba,
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
                
                try:
                    #test += 2
                #except NameError:
                    k_accuracys_local_surrogate, growing_field_radius = explainer.explain_instance(instance_to_explain, 
                                                            growing_method="GF", k_radius_local_surrogate=k_radius_local_surrogate)

                    
                    k_accuracys_local_surrogate += [growing_field_radius, explainer.friends_pvalue, explainer.counterfactual_pvalue,
                                                            explainer.separability_index, explainer.friends_folding_statistics,
                                                            explainer.counterfactual_folding_statistics, model_name, dataset_name]
                    
                    if graph:
                            experimental_informations.store_experiments_information_instance(k_accuracys_local_surrogate, \
                                'k_radius_local_surrogate_extended.csv')
                    cnt += 1
                    if cnt %5 == 0:
                        print()
                        print("### Instance number:", cnt , "over", str(max_instance_to_explain), 'with', model_name, 'on', dataset_name)

                #except Exception as inst:
                    #print(inst)
                except TypeError:
                    print()
            if graph: experimental_informations.store_experiments_information('k_radius_local_surrogate_extended.csv', filename_all=filename_all)