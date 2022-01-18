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
    dataset_names = ["generate_circles", "generate_moons", "blood", "diabetes", "generate_blobs"]
    # array of the models used for the experiments
    models = [GradientBoostingClassifier(n_estimators=20, learning_rate=1.0),
                RandomForestClassifier(n_estimators=20), 
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('rc', LogisticRegression())], voting="soft"),
                MLPClassifier(random_state=1),
                RidgeClassifier()]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    # Print explanation result
    illustrative_example = False
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    growing_sphere = False
    growing_method = "GS" if growing_sphere else "GF"

    # Threshold for explanation method precision
    threshold_interpretability = 0.95
    linear_separability_index = 1
    interpretability_name = ['ls', 'ls regression', 'ls raw data', 'ls extend']
    #interpretability_name = ['ls log reg', 'ls raw data']
    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, \
                    feature_names, transformations = generate_dataset(dataset_name)
        
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            if growing_sphere:
                filename = "./results/"+dataset_name+"/"+model_name+"/growing_spheres/"+str(threshold_interpretability)+"/sup_mat_"
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/sup_mat_"
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/sup_mat_"
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/sup_mat_"
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name)
            black_box = black_box.fit(x_train, y_train)
            print("###", model_name, "training on", dataset_name, "dataset.")
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, black_box.predict, black_box.predict_proba,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=feature_names, categorical_names=categorical_names,
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
                    test+=2
                except:
                    accuracy, coverage, f2 = explainer.explain_instance(instance_to_explain, 
                                                    growing_method=growing_method, 
                                                    local_surrogate_experiment=True)
                    print("accuracy", accuracy)
                    print("coverage", coverage)
                    print("f2", f2)
                    if graph: experimental_informations.store_experiments_information_instance(accuracy, 'accuracy.csv', coverage, 'coverage.csv', f2, 'f2.csv')
                    cnt += 1
                #except Exception as inst:
                    #print(inst)

            if graph: experimental_informations.store_experiments_information('accuracy.csv', 'coverage.csv', 'f2.csv', filename_all=filename_all)
            