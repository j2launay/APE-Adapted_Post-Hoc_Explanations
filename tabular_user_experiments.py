from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from storeExperimentalInformations import store_experimental_informations
from generate_dataset import generate_dataset, preparing_dataset, modify_dataset
import ape_tabular
import warnings
import random
import scipy.stats as stats
from ape_experiments_functions import decision_tree_function

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
    dataset_names = ["generate_blobs", "compas", "adult", "mega_generate_blobs", "diabetes", "generate_blobs", "cancer"]
    # Models employed for experiments
    models = [tree.DecisionTreeClassifier(random_state=1), 
            RandomForestClassifier(n_estimators=20, random_state=1), 
            LogisticRegression(random_state=1), 
            GradientBoostingClassifier(n_estimators=20, random_state=1)]

    # Number of instance for which explanations are computed
    max_instance_to_explain = 100
    # Number of feature from the dataset that are modified (values are set to 0 to train the decision model)
    nb_feature_to_train = 8
    # If set to True store the results inside a graph
    graph = True
    # If set to True print detailed information
    verbose = False
    # Precision threshold for explanation models and linear separability test 
    threshold_interpretability = 0.99
    interpretability_name = ['LS prec', 'LS pos prec', 'LIME prec', 'LIME pos prec', 'LSe prec', 'LSe pos prec', 'LSe2 prec', 'Anc. prec', 
                                    'rand. prec', 'APE prec', 'LS reca', 'LS pos reca', 'LIME reca', 'LIME pos reca', 'LSe reca', 'LSe pos reca', 
                                    'LSe2 reca', 'Anc. reca', 'rand. reca', 'APE reca', "Multimodal", "radius", "fr pvalue", "cf pvalue", 
                                    "separability", "fr fold", "cf fold", "bb", "dataset"]
    kendall_tau_name = ['LS', 'LS pos', 'LIME', 'LIME pos', 'LSe', 'LSe pos', 'LSe2', 'Anc.', 'rand.', 'APE',
                                    'LS pvalue', 'LS pos pvalue', 'LIME pvalue', 'LIME pos pvalue', 'LSe pvalue', 'LSe pos pvalue', 'LSe2 pvalue', 
                                    'Anc. pvalue', 'rand. pvalue', 'APE pvalue', "Multimodal", "radius", 
                                    "fr pvalue", "cf pvalue", "separability", "fr fold", "cf fold", "bb", "dataset"]
    # Initialize variable to store the results for the graph representation
    for dataset_name in dataset_names:
        if graph: experimental_informations = store_experimental_informations(interpretability_name, kendall_tau_name)
        # Store dataset information such as class names and the list of categerical features as well as variables (x for input and y for labels)
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, \
            categorical_names, feature_names, transformations = generate_dataset(dataset_name)
        
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            filename = "./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.initialize_per_models(filename)
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y)
            print("###", model_name, "training on", dataset_name, "dataset.")
            # Modify the dataset to train the "black box" model only on a subset of features
            nb_feature_to_modify = int(len(x_train[0]) / 2)# nb_feature_to_train
            #print("nb feature to train", nb_feature_to_train)
            print("nb feature to modify", nb_feature_to_modify)
            x_train_bb, feature_kept = modify_dataset(x_train, nb_feature_to_modify)
            black_box = black_box.fit(x_train_bb, y_train)
            
            if "Tree" in model_name or 'Forest' in model_name or 'Gradient' in model_name:# and verbose:
                print("features importances", black_box.feature_importances_)
            elif verbose:
                print("features importances", black_box.coef_)            

            print('### Accuracy:', sum(black_box.predict(x_test) == y_test)/len(y_test))
            cnt = 0
            explainer = ape_tabular.ApeTabularExplainer(x_test, class_names, black_box.predict, black_box.predict_proba, continuous_features=continuous_features, 
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=feature_names, categorical_names=categorical_names,
                                                            verbose=False, linear_separability_index=1, transformations=transformations)
            
            for instance_to_explain in x_test: 
                if cnt == max_instance_to_explain:
                    break
                print("### Models ", nb_model+1, "over", len(models))
                print("### Instance number:", cnt+1, "over", max_instance_to_explain)
                if "Tree" in model_name:
                    # Store the features that were actually used by the decision tree model to classify instances
                    features_employed_black_box = decision_tree_function(black_box, instance_to_explain.reshape(1, -1))
                else:
                    # Store the features that were actually used by the Logistic Regression model to classify instances
                    features_employed_black_box = feature_kept

                if verbose: print("features employed by the black box", features_employed_black_box)
                
                try:
                    # Get the list of features employed by the 3 explanation models 
                    features_employed_in_local_surrogate, pos_feat_ls, feat_lime, pos_feat_lime, feat_lse, pos_feat_lse, feat_lse2, \
                        features_employed_by_ape, features_employed_anchors,\
                        multimodal_result, radius = explainer.explain_instance(instance_to_explain,
                                                                    user_experiments=True, 
                                                                    nb_features_employed=len(features_employed_black_box))
                    # Selects randomly as many features as the black box model is actually chosen among all the features
                    random_explainer = random.sample(range(len(instance_to_explain)), len(features_employed_black_box))
                    
                    precision_local_surrogate, recall_local_surrogate = compute_score_interpretability_method(features_employed_in_local_surrogate, features_employed_black_box)
                    kendall_ls, pvalue_ls = stats.kendalltau(features_employed_in_local_surrogate[:len(features_employed_black_box)], features_employed_black_box[:len(features_employed_in_local_surrogate)])
                    precision_pos_ls, recall_pos_ls = compute_score_interpretability_method(pos_feat_ls, features_employed_black_box)
                    kendall_pos_ls, pvalue_pos_ls = stats.kendalltau(pos_feat_ls[:len(features_employed_black_box)], features_employed_black_box[:len(pos_feat_ls)])
                    precision_lime, recall_lime = compute_score_interpretability_method(feat_lime, features_employed_black_box)
                    kendall_lime, pvalue_lime = stats.kendalltau(feat_lime[:len(features_employed_black_box)], features_employed_black_box[:len(feat_lime)])
                    precision_pos_lime, recall_pos_lime = compute_score_interpretability_method(pos_feat_lime, features_employed_black_box)
                    kendall_pos_lime, pvalue_pos_lime = stats.kendalltau(pos_feat_lime[:len(features_employed_black_box)], features_employed_black_box[:len(pos_feat_lime)])
                    precision_lse, recall_lse = compute_score_interpretability_method(feat_lse, features_employed_black_box)
                    kendall_lse, pvalue_lse = stats.kendalltau(feat_lse[:len(features_employed_black_box)], features_employed_black_box[:len(feat_lse)])
                    precision_pos_lse, recall_pos_lse = compute_score_interpretability_method(pos_feat_lse, features_employed_black_box)
                    kendall_pos_lse, pvalue_pos_lse = stats.kendalltau(pos_feat_lse[:len(features_employed_black_box)], features_employed_black_box[:len(pos_feat_lse)])
                    precision_lse2, recall_lse2 = compute_score_interpretability_method(feat_lse2, features_employed_black_box)
                    kendall_lse2, pvalue_lse2 = stats.kendalltau(feat_lse2[:len(features_employed_black_box)], features_employed_black_box[:len(feat_lse2)])
                    precision_ape, recall_ape = compute_score_interpretability_method(features_employed_by_ape, features_employed_black_box)
                    kendall_ape, pvalue_ape = stats.kendalltau(features_employed_by_ape[:len(features_employed_black_box)], features_employed_black_box[:len(features_employed_by_ape)])
                    precision_anchor, recall_anchor = compute_score_interpretability_method(features_employed_anchors, features_employed_black_box)
                    kendall_anchor, pvalue_anchor = stats.kendalltau(features_employed_anchors[:len(features_employed_black_box)], features_employed_black_box[:len(features_employed_anchors)])
                    precision_random, recall_random = compute_score_interpretability_method(random_explainer, features_employed_black_box)
                    kendall_random, pvalue_random = stats.kendalltau(random_explainer[:len(features_employed_black_box)], features_employed_black_box[:len(random_explainer)])
                    cnt += 1
                    
                    if graph: 
                        experimental_informations.store_experiments_information_instance([precision_local_surrogate, precision_pos_ls, precision_lime, 
                            precision_pos_lime, precision_lse, precision_pos_lse, precision_lse2, precision_anchor, precision_random, precision_ape, 
                                recall_local_surrogate, recall_pos_ls, recall_lime, recall_pos_lime, recall_lse, recall_pos_lse, recall_lse2, 
                                    recall_anchor, recall_random, recall_ape, multimodal_result, radius, explainer.friends_pvalue, 
                                        explainer.counterfactual_pvalue, explainer.separability_index, explainer.friends_folding_statistics,
                                            explainer.counterfactual_folding_statistics, model_name, dataset_name], 'user_experiments.csv',
                            [kendall_ls, kendall_pos_ls, kendall_lime, kendall_pos_lime, kendall_lse, kendall_pos_lse, kendall_lse2, 
                                kendall_anchor, kendall_random, kendall_ape, pvalue_ls, pvalue_pos_ls, pvalue_lime, pvalue_pos_lime, 
                                    pvalue_lse, pvalue_pos_lse, pvalue_lse2, pvalue_anchor, pvalue_random, pvalue_ape, 
                                         multimodal_result, radius, explainer.friends_pvalue, 
                                            explainer.counterfactual_pvalue, explainer.separability_index, explainer.friends_folding_statistics,
                                                explainer.counterfactual_folding_statistics, model_name, dataset_name], 'user_experiments_kendall.csv')
                    if cnt %5 == 0:
                        print()
                        print("### Instance number:", cnt , "over", max_instance_to_explain, 'with', model_name, 'on', dataset_name)

                except Exception as inst:
                    print(inst)
                    
            filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/"
            if graph: experimental_informations.store_experiments_information('user_experiments.csv', filename_all=filename_all)
