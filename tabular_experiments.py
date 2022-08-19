from tokenize import Name
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
from growingspheres.utils.gs_utils import distances
import ape_tabular
import warnings

def evaluate_test_unimodality(bool, accuracy_ls, accuracy_anchor):
    if bool:
        if accuracy_ls >= accuracy_anchor:
            return 1
        else:
            return 0
    else:
        if accuracy_ls <= accuracy_anchor:
            return 1
        else:
            return 0

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments 
    dataset_names = ["generate_moons", "generate_blob", "generate_circles", "generate_moons", 'categorical_generate_blobs', "generate_blobs", "blood", \
        "cancer", "compas", "adult", "diabetes", 'titanic']

    models = [GaussianNB(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('svm', svm.SVC(probability=True))], voting='soft'),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                RandomForestClassifier(n_estimators=20, random_state=1), 
                MLPClassifier(random_state=1, activation='logistic'),
                svm.SVC(probability=True, random_state=1, class_weight="balanced")]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 100
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    growing_method = "GF"

    # Threshold for explanation method precision
    threshold_interpretability = 0.95
    linear_separability_index = 1
    interpretability_name = ['LS', 'LS roc', 'LSe ini', 'LSe ini auc', 'LSe', 'LSe auc', 'LIME', 'LIME roc', 'Anchors', 'DT', 'APEa', 'APEt']
    linear_model = Ridge(alpha=0)#SGDRegressor()#LinearRegression()#
    linear_model_name = type(linear_model).__name__ + "_" if "Ridge" not in type(linear_model).__name__ else ""

    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(interpretability_name)
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, \
                    feature_names, transformations = generate_dataset(dataset_name)
        
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/" + linear_model_name
            filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/" + linear_model_name
            if graph: experimental_informations.initialize_per_models(filename)
            models_name.append(model_name)
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y)
            print("###", model_name, "training on", dataset_name, "dataset.")
            print()
            print()
            black_box = black_box.fit(x_train, y_train)
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
                    accuracy, coverage, f2, multimodal_result, radius = explainer.explain_instance(instance_to_explain, 
                                                    growing_method=growing_method, 
                                                    linear_model=linear_model,
                                                    all_explanations_model=True)
                    print("accuracy", accuracy)
                    
                    if graph:
                        experimental_informations.store_experiments_information_instance(accuracy, 'accuracy.csv', coverage, 
                                                    'coverage.csv', f2, 'f2.csv',
                                                    multimodal=[accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4],
                                                                                    accuracy[5], accuracy[6], accuracy[7], accuracy[8], 
                                                                                    accuracy[9], accuracy[10], accuracy[11], multimodal_result, 
                                                                                    radius, explainer.friends_pvalue, 
                                                                                    explainer.counterfactual_pvalue,
                                                                                    explainer.separability_index, explainer.friends_folding_statistics,
                                                                                    explainer.counterfactual_folding_statistics, model_name, dataset_name])
                    cnt += 1
                except Exception as inst:
                    print(inst)
                if cnt %5 == 0:
                    print()
                    print("### Instance number:", cnt , "over", max_instance_to_explain, 'with', model_name, 'on', dataset_name)

            if graph: experimental_informations.store_experiments_information(filename1='accuracy.csv', 
                    filename2='coverage.csv', filename3='f2.csv', filename_multimodal="multimodal.csv", 
                    filename_all=filename_all)

            