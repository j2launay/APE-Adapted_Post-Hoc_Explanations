from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import generate_dataset, preparing_dataset
from storeExperimentalInformations import store_experimental_informations
import ape_tabular
import warnings
#from keras.models import Sequential
#from keras.layers import Dense

def evaluate_test_unimodality(bool, precision_ls, precision_anchor):
    if bool:
        if precision_ls >= precision_anchor:
            return 1
        else:
            return 0
    else:
        if precision_ls <= precision_anchor:
            return 1
        else:
            return 0

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments
    # "generate_circles", "generate_moons", "generate_blob", "diabete", "generate_blobs", 
    dataset_names = ['titanic']#["adult", "compas", "titanic", "mortality", 'categorical_generate_blobs', "blood"]
    # array of the models used for the experiments
    models = [VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('svm', svm.SVC(probability=True))], voting='soft'),#('rc', RidgeClassifier())], voting="soft"),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                #MLPClassifier(random_state=1, activation="logistic"),
                RandomForestClassifier(n_estimators=20, random_state=1), 
                MLPClassifier(random_state=1),
                svm.SVC(probability=True, random_state=1)]
                #RidgeClassifier(random_state=1)]
                #Sequential(),
                #GaussianNB
                #KNeighborsClassifier
                #LinearDiscriminantAnalysis
    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 30
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
    linear_models_name = ['local surrogate', 'lime extending', 'lime regression', 'lime not binarize', 'lime traditional']
    interpretability_name = ['LS', 'LSe log', 'LSe lin', 'Anchors', 'APE SI', 'APE CF', 'APE FOLD', 'APE FULL', 'APE FULL pvalue', 'DT']
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
            # Split the dataset inside train and test set (50% each set)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y, dataset_name)
            print()
            print()
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
                #print("class", black_box.predict_proba(instance_to_explain.reshape(1, -1))[0])

                try:
                    #test+=2
                #except:
                    precision, coverage, f2, multimodal_result, radius, real_precisions = explainer.explain_instance(instance_to_explain, 
                                                    growing_method=growing_method, 
                                                    all_explanations_model=True)
                    print("precision", precision)
                    print("real precision", real_precisions)
                    """print("coverage", coverage)
                    print("f2", f2)
                    
                    print("multimodal", multimodal_result)
                    print("radius", radius)
                    print("separability", explainer.separability_index)
                    print("friends pvalue", explainer.friends_pvalue)
                    print("counterfactual pvalue", explainer.counterfactual_pvalue)
                    print("friends folding statistic", explainer.friends_folding_statistics)
                    print("counterfactual folding statistic", explainer.counterfactual_folding_statistics)
                    # Evaluate whether the linear separability index returns truely the case where the precision of LS is better than Anchors
                    si_bon = evaluate_test_unimodality(explainer.separability_index >= linear_separability_index, precision[1], precision[2])
                    # Evaluate whether the unimodality test returns truely the case where the precision of LS is better than Anchors
                    fold_bon = evaluate_test_unimodality(explainer.friends_folding_statistics >= 1 and explainer.counterfactual_folding_statistics >=1,  
                                                                        precision[1], precision[3])
                    cf_bon = evaluate_test_unimodality(explainer.counterfactual_folding_statistics >=1, precision[1], )
                    """
                    cf_bon = 1 if (precision[5] >= precision[1] and precision[5] >= precision[3]) else 0
                    ape_bon = 1 if (precision[4] >= precision[1] and precision[4] >= precision[3]) else 0
                    si_bon = 1 if (precision[7] >= precision[1] and precision[7] >= precision[3]) else 0
                    fold_bon = 1 if (precision[6] >= precision[1] and precision[6] >= precision[3]) else 0
                    ape_pvalue_bon = 1 if (precision[8] >= precision[1] and precision[8] >= precision[3]) else 0
                    print("separability index bon", si_bon)
                    print("counterfactual folding bon", cf_bon)
                    print("fold bon", fold_bon)
                    print("ape bon", ape_bon)
                    print("ape pvalue bon", ape_pvalue_bon)
                    if graph: experimental_informations.store_experiments_information_instance(precision, 'precision.csv', coverage, 
                                                    'coverage.csv', f2, 'f2.csv', real_precisions, 'real_precisions.csv',
                                                    multimodal=[precision[0], precision[1], precision[2], precision[3], precision[4], precision[5],
                                                                                    precision[6], precision[7], precision[8], precision[9],
                                                                                    multimodal_result, radius, explainer.friends_pvalue, 
                                                                                    explainer.counterfactual_pvalue,
                                                                                    explainer.separability_index, explainer.friends_folding_statistics,
                                                                                    explainer.counterfactual_folding_statistics, si_bon, cf_bon,
                                                                                    fold_bon, ape_bon, ape_pvalue_bon, model_name])
                    cnt += 1
                except Exception as inst:
                    print(inst)

            if graph: experimental_informations.store_experiments_information(max_instance_to_explain, nb_model, filename1='precision.csv', 
                    filename2='coverage.csv', filename3='f2.csv', filename4='real_precisions.csv', filename_multimodal="multimodal.csv", 
                    filename_all=filename_all)

            