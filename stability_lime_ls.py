from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
import warnings
import ape_tabular
from lime_vs_local_surrogate import find_closest_counterfactual, get_farthest_distance

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_blob", "generate_moons", "blood", "diabete"]# Fais jusqu'ici, "titanic", "compas", "adult"]
    # array of the models used for the experiments
    models = [VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), 
                                            ('mlp', MLPClassifier(random_state=1, activation="logistic"))], voting="soft"),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                #Sequential(),
                MLPClassifier(random_state=1, activation="logistic"),
                RandomForestClassifier(n_estimators=20, random_state=1), 
                MLPClassifier(random_state=1)]#,
                #LogisticRegression(),
                #tree.DecisionTreeClassifier(),
                
    #models=[RidgeClassifier(), MLPClassifier(random_state=1)]
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 25
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
    linear_separability_index = 1
    interpretability_name = ['LIME', 'LS', 'LSe']
    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, \
                    categorical_values, categorical_names, transformations = generate_dataset(dataset_name)
        for nb_model, model in enumerate(models):
            model_name = type(model).__name__
            if "MLP" in model_name and nb_model <=2 :
                model_name += "logistic"
            if growing_sphere:
                filename = "./results/"+dataset_name+"/growing_spheres/"+model_name+"/"+str(threshold_interpretability)+"/"
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
            dataset, black_box, x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name, model)
            print("###", model_name, "training on", dataset_name, "dataset.")
            black_box = black_box.fit(x_train, y_train)
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
            
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, black_box.predict, #black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=dataset.feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability,
                                                            linear_separability_index=linear_separability_index, 
                                                            transformations=transformations)
            
            
            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
            
                try:
                    # Search for the closest counterfactual in order to compute the csi and vsi of the linear explanation over the closest counterfactual
                    closest_counterfactual, radius = find_closest_counterfactual(instance_to_explain, explainer, method='GS', radius=True)
                    farthest_distance = get_farthest_distance(instance_to_explain, x_train, categorical_features, explainer, metric='manhattan')
                    
                    csi_lime, vsi_lime = explainer.linear_explainer.check_stability(instance_to_explain, black_box.predict_proba, 
                                            n_calls=10, index_verbose=False, ls=False, ape=explainer)

                    csi_ls, vsi_ls = explainer.linear_explainer.check_stability(closest_counterfactual, black_box.predict_proba, 
                                            n_calls=10, index_verbose=False, ls=False, ape=explainer)
                    
                    # Generate artificial instances in the fields close to the closest counterfactual to compute the Extended Local Surrogate
                    explainer.target_class = black_box.predict(instance_to_explain.reshape(1, -1))
                    position_instances_in_field, nb_training_instance_in_field = explainer.instances_from_dataset_inside_field(
                                            closest_counterfactual, radius, x_train)
                    instances_in_field = explainer.generate_instances_inside_field(radius, closest_counterfactual, x_train, 
                                        farthest_distance, 100, position_instances_in_field, nb_training_instance_in_field, growing_method="GF", 
                                            libfolding=False, lime_ls=False)
                    csi_lse, vsi_lse = explainer.linear_explainer.check_stability(closest_counterfactual, black_box.predict_proba, 
                                            n_calls=10, index_verbose=False, ls=False, ape=explainer, instances_in_field=instances_in_field)
                    print("csi LIME", csi_lime)
                    print("vsi LIME", vsi_lime)
                    print("csi LS", csi_ls)
                    print("vsi LS", vsi_ls)
                    print("csi LSe", csi_lse)
                    print("vsi LSe", vsi_lse)
                    
                    if graph: experimental_informations.store_experiments_information_instance(results1=[csi_lime, csi_ls, csi_lse], 
                                    results2=[vsi_lime, vsi_ls, vsi_lse], filename1="csi_ls.csv", filename2="vsi_ls.csv")
                    cnt += 1
                except Exception as inst:
                    print(inst)

            if graph: experimental_informations.store_experiments_information(filename1='csi_ls.csv', 
                                    filename2='vsi_ls.csv', filename_all=filename_all)
           