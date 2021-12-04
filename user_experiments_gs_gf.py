from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from storeExperimentalInformations import store_experimental_informations
from generate_dataset import generate_dataset, preparing_dataset
from sklearn.model_selection import train_test_split
import ape_tabular
import warnings
import random
from ape_tabular_experiments import modify_dataset, decision_tree_function, user_experiments_gs_gf

def compute_score_interpretability_method(features_employed_by_explainer, features_employed_black_box):
    """
    Compute the score of the explanation method based on the features employed for the explanation compared to the features truely used by the black box
    """
    precision = 0
    recall = 0
    for feature_employe in features_employed_by_explainer:
        if feature_employe in features_employed_black_box:
            precision += 1
    for feature_employe in features_employed_black_box:
        if feature_employe in features_employed_by_explainer:
            recall += 1
    return precision/len(features_employed_by_explainer), recall/len(features_employed_black_box)

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # The datasets employed for experiments
    dataset_names = ["diabete", "titanic", "generate_blobs", "mega_generate_blobs"]
    # Models employed for experiments
    models = [RandomForestClassifier(n_estimators=20, random_state=1), tree.DecisionTreeClassifier(random_state=1), LogisticRegression(random_state=1)]# tree.DecisionTreeClassifier(max_depth=4)]
    models_name = ['RandomForestClassifier', 'DecisionTreeClassifier', 'LogisticRegression']#, 'DecisionTreeClassifier_depth4']    
    # Number of instance for which explanations are computed
    max_instance_to_explain = 50
    # Number of feature from the dataset that are modified (values are set to 0 to train the decision model)
    nb_feature_to_train = 6 
    # If set to True store the results inside a graph
    graph = True
    # If set to True print detailed information
    verbose = False
    # Precision threshold for explanation models and linear separability test 
    threshold_interpretability = 0.99
    interpretability_name = ['local surrogate', 'anchors', 'random', 'ape', 'local surrogate', 'anchors', 'random', 'ape']
    interpretability_name = ['LS GS prec', 'LS GF prec', 'Anc GS prec', 'Anc GF prec', 'rand. prec',  'LS GS reca', 'LS GF reca', 
                                    'Anc GS reca', 'Anc GF reca', 'rand. reca']
    # Initialize variable to store the results for the graph representation
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(len(models), len(interpretability_name), interpretability_name, len(models))
        # Store dataset information such as class names and the list of categerical features as well as variables (x for input and y for labels)
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, features_name, transformations = generate_dataset(dataset_name)
        for nb_model, black_box in enumerate(models):
            model_name = models_name[nb_model]
            filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            # Split the dataset in test and train set (50% each)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
            print("###", model_name, "training on", dataset_name, "dataset.")
            # Modify the dataset to train the "black box" model only on a subset of features
            nb_feature_to_modify = len(x_train[0]) - nb_feature_to_train
            print("nb feature to train", nb_feature_to_train)
            print("nb feature to modify", nb_feature_to_modify)
            x_train_bb, feature_kept = modify_dataset(x_train, nb_feature_to_modify)
            black_box = black_box.fit(x_train_bb, y_train)
            
            if "Tree" in model_name or "Forest" in model_name:# and verbose:
                print("features importances", black_box.feature_importances_)
            elif verbose:
                print("features importances", black_box.coef_)            

            #print("feature employed by black box", features_employed_black_box)
            print('### Accuracy:', sum(black_box.predict(x_test) == y_test)/len(y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_test, class_names, black_box.predict, black_box.predict_proba, continuous_features=continuous_features, 
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=features_name, categorical_names=categorical_names,
                                                            verbose=False, linear_separability_index=1, transformations=transformations)
            for instance_to_explain in x_test: 
                if cnt == max_instance_to_explain:
                    break
                print("### Models ", nb_model+1, "over", len(models))
                print("### Instance number:", cnt+1, "over", max_instance_to_explain)
                print(instance_to_explain)
                if "Tree" in model_name:
                    # Store the features that were actually used by the decision tree model to classify instances
                    features_employed_black_box = decision_tree_function(black_box, instance_to_explain.reshape(1, -1))
                else:
                    # Store the features that were actually used by the Logistic Regression model to classify instances
                    features_employed_black_box = feature_kept

                if verbose: print("features employed by the black box", features_employed_black_box)
                
                try:
                    # Get the list of features employed by the 3 explanation models 
                    features_employed_in_ls_gs, features_employed_in_ls_gf, features_employed_anchors_gs, features_employed_anchors_gf = \
                                                                    user_experiments_gs_gf(instance_to_explain, explainer, 
                                                                    nb_features_employed=len(features_employed_black_box))
                    # Selects randomly as many features as the black box model is actually chosen among all the features
                    random_explainer = random.sample(range(len(instance_to_explain)), len(features_employed_black_box))
                    
                    #if verbose:
                        #print("features employed by Local Surrogate", features_employed_in_local_surrogate)
                        #print("features employed by APE", features_employed_by_ape)
                        #print("features employed by anchors", features_employed_anchors)
                        #print("features employed randomly", random_explainer)
                    print("features employed by the black box", features_employed_black_box)
                    print("features employed by Local Surrogate GS", features_employed_in_ls_gs)
                    print("features employed by Local Surrogate GF", features_employed_in_ls_gf)
                    print("features employed by anchors GS", features_employed_anchors_gs)
                    print("features employed by anchors GF", features_employed_anchors_gf)
                    print("features employed randomly", random_explainer)
                    precision_ls_gs, recall_ls_gs = compute_score_interpretability_method(features_employed_in_ls_gs, 
                                                        features_employed_black_box)
                    precision_ls_gf, recall_ls_gf = compute_score_interpretability_method(features_employed_in_ls_gf, 
                                                        features_employed_black_box)
                    
                    precision_anchor_gf, recall_anchor_gf = compute_score_interpretability_method(features_employed_anchors_gf, features_employed_black_box)
                    precision_anchor_gs, recall_anchor_gs = compute_score_interpretability_method(features_employed_anchors_gs, features_employed_black_box)
                    precision_random, recall_random = compute_score_interpretability_method(random_explainer, features_employed_black_box)
                    cnt += 1

                    if graph: experimental_informations.store_experiments_information_instance([precision_ls_gs, precision_ls_gf,
                                    precision_anchor_gs, precision_anchor_gf, precision_random, recall_ls_gs, recall_ls_gf, recall_anchor_gs,
                                    recall_anchor_gf, recall_random], 'recall_gs_gf.csv')
                except Exception as inst:
                    print(inst)
                    
            filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, 'recall_gs_gf.csv', filename_all=filename_all)
