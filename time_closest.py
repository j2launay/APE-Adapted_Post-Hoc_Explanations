from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
import ape_tabular
import warnings
import time

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    dataset_names = ["generate_moons", "blood", "diabetes", "generate_circles", "generate_blob", "mega_generate_blobs", "generate_blobs"]
    # array of the models used for the experiments
    models = [RandomForestClassifier(n_estimators=20, random_state=1),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                RidgeClassifier(random_state=1),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('rc', RidgeClassifier())], voting="hard"),
                MLPClassifier(random_state=1)]


    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False

    # Threshold for explanation method precision
    threshold_interpretability = 0.99
    interpretability_name = ['Growing Spheres', 'Growing Fields']
    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(interpretability_name)
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, \
            categorical_names, feature_names, transformations = generate_dataset(dataset_name)
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y)
            print("###", model_name, "training on", dataset_name, "dataset.")
        
            black_box = black_box.fit(x_train, y_train)
            predict = black_box.predict
            score = black_box.score
            print('### Accuracy:', score(x_test, y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_train, class_names, predict,
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
                
                start_time = time.time()
                explainer.explain_instance(instance_to_explain, time_k_closest=True, growing_method="GS")
                time_gs = time.time() - start_time
                
                start_time = time.time()
                explainer.explain_instance(instance_to_explain, time_k_closest=True, growing_method="GF")
                time_gf = time.time() - start_time

                print("average time for GF", time_gf)
                print("average time for GS", time_gs)
                print("model", model_name)
                print("dataset", dataset_name)
                try:
                    print("GF", "better" if time_gf < time_gs else "worse", "than GS")
                    if graph: 
                        experimental_informations.store_experiments_information_instance([time_gs, time_gf], 'average_time.csv')
                    cnt += 1
                except Exception as inst:
                    print(inst)
            if graph: experimental_informations.store_experiments_information('average_time.csv',
                                                                        filename_all=filename_all)