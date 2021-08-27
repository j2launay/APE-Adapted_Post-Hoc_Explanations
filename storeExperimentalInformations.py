import numpy as np
import pandas as pd
import csv
import os

def prepare_legends(mean_models, models, interpretability_name):
    bars = []
    y_pos = []
    index_bars = 0
    for nb, i in enumerate(mean_models):
        if nb % len(models) == int(len(models)/2):
            bars.append(interpretability_name[index_bars])
            index_bars += 1
        else:
            bars.append('')
        if nb < len(mean_models)/len(interpretability_name):
            y_pos.append(nb)
        elif nb < 2*len(mean_models)/len(interpretability_name):
            y_pos.append(nb+1)
        elif nb < 3*len(mean_models)/len(interpretability_name):
            y_pos.append(nb+2)
        elif nb < 4*len(mean_models)/len(interpretability_name):
            y_pos.append(nb+3)
        elif nb < 5*len(mean_models)/len(interpretability_name):
            y_pos.append(nb+4)
        else:
            y_pos.append(nb+5)
        
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'grey', 'purple', 'cyan', 'gold', 'brown']
    color= []
    for nb, model in enumerate(models):
        color.append(colors[nb])
    return color, bars, y_pos

class store_experimental_informations(object):
    """
    Class to store the experimental results of precision, coverage and F1 score for graph representation
    """
    def __init__(self, len_models, len_interpretability_name, interpretability_name, nb_models):
        """
        Initialize all the variable that will be used to store experimental results
        Args: len_models: Number of black box models that we are explaining during experiments
              len_interpretability_name: Number of explanation methods used to explain each model
              interpretability_name: List of the name of the explanation methods used to explain each model 
        """
        self.interpretability_name = interpretability_name
        self.len_interpretability_name, self.len_models, = len_interpretability_name, len_models
        size_results = len_models*len_interpretability_name
        self.final_precisions, self.final_coverages, self.final_f2s = np.zeros(size_results), np.zeros(size_results), np.zeros(size_results)
        self.final_recalls = np.zeros(size_results)
        self.final_multimodals = np.zeros(len_models)
        self.nb_models = nb_models - 1
        self.pd_all_models_precision = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_coverage = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_f2 = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_stability = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_stability_features = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_msi = pd.DataFrame(columns=['APE'])
        self.pd_all_models_csi = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_vsi = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_recall = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_distance = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_lime_ls = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_degrees = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_kendall = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_top_k = pd.DataFrame(columns=interpretability_name)
        self.pd_all_models_counterfactual_in_anchor = pd.DataFrame(columns=interpretability_name)

    def initialize_per_models(self, filename):
        self.precision = {}
        self.coverage = {}
        self.f2 = {}
        self.recall_user_experiments = {}
        self.multimodal = []
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename+"/"), exist_ok=True)
        self.pd_stability = pd.DataFrame(columns=self.interpretability_name)
        self.pd_stability_features = pd.DataFrame(columns=self.interpretability_name)
        self.pd_msi = pd.DataFrame(columns=['APE'])
        self.pd_csi = pd.DataFrame(columns=self.interpretability_name)
        self.pd_vsi = pd.DataFrame(columns=self.interpretability_name)

        self.pd_precision = pd.DataFrame(columns=self.interpretability_name)
        self.pd_coverage = pd.DataFrame(columns=self.interpretability_name)
        self.pd_f2 = pd.DataFrame(columns=self.interpretability_name)
        self.pd_recall = pd.DataFrame(columns=self.interpretability_name)
        self.pd_average_distance = pd.DataFrame(columns=self.interpretability_name)
        self.pd_lime_ls = pd.DataFrame(columns=self.interpretability_name)
        self.pd_degrees = pd.DataFrame(columns=self.interpretability_name)
        self.pd_kendall = pd.DataFrame(columns=self.interpretability_name)
        self.pd_counterfactual_in_anchor = pd.DataFrame(columns=self.interpretability_name)
        for interpretability in self.interpretability_name:
            self.precision[interpretability] = []
            self.coverage[interpretability] = []
            self.f2[interpretability] = []
            self.recall_user_experiments[interpretability] = []

    def store_experiments_information_instance(self, precisions, coverages, f2s, multimodal=None):
        """
        Store precisions, coverages, f2s and multimodal results inside dictionary 
        Args: precisions: list of precision result for each explanation method on a single instance
              coverages: list of coverage result for each explanation method on a single instance
              f2s: list of f2 score for each explanation method on a single instance
              multimodal: 1 if APE selected a multimodal distribution, otherwise 0
        """
        self.pd_precision = self.pd_precision.append(pd.DataFrame([precisions], columns=self.interpretability_name))
        self.pd_coverage = self.pd_coverage.append(pd.DataFrame([coverages], columns=self.interpretability_name))
        self.pd_f2 = self.pd_f2.append(pd.DataFrame([f2s], columns=self.interpretability_name))
        self.pd_precision.to_csv(self.filename + 'precision.csv', index=False)
        self.pd_coverage.to_csv(self.filename + 'coverage.csv', index=False)
        self.pd_f2.to_csv(self.filename + 'f2.csv', index=False)
        for precision, coverage, f2, interpretability in zip(precisions, coverages, f2s, self.interpretability_name):
            if self.precision[interpretability] == []:
                self.precision[interpretability] = precision
                self.coverage[interpretability] = coverage
                self.f2[interpretability] = f2
            else:
                self.precision[interpretability] += precision
                self.coverage[interpretability] += coverage
                self.f2[interpretability] += f2
        if multimodal is not None:
            if self.multimodal == []:
                self.multimodal = multimodal
            else:
                self.multimodal += multimodal

    def store_experiments_information(self, nb_instance, nb_model, filename_all=""):
        """ 
        Compute the mean coverage, precision and f2 per model 
        Args: nb_instance: Number of instance for which we generate explanation for each model
              nb_model: Numerous of the black box model for which we generate explanation (first model employed = 0 , second model employed = 1, etc...)
        """
        os.makedirs(os.path.dirname(self.filename+"/"), exist_ok=True)
        os.makedirs(os.path.dirname(filename_all+"/"), exist_ok=True)

        self.final_precision = []
        self.final_coverage = []
        self.final_f2 = []
        for interpretability in self.interpretability_name:
            if not self.precision[interpretability] == []:
                # For each explanation method we compute the mean precision, coverage and f2 for graph displaying
                self.final_precision.append(self.precision[interpretability] / nb_instance)
                self.final_coverage.append(self.coverage[interpretability] / nb_instance)
                self.final_f2.append(self.f2[interpretability] / nb_instance)
        if not self.final_precision == []: 
            for nb, interpretability in enumerate(self.interpretability_name):
                # Store in arrays the mean precision, coverage and f2 for graph displaying
                self.final_precisions[nb*self.len_models + nb_model] = self.final_precision[nb]
                self.final_coverages[nb*self.len_models + nb_model] = self.final_coverage[nb]
                self.final_f2s[nb*self.len_models + nb_model] = self.final_f2[nb]
            self.pd_all_models_precision = self.pd_all_models_precision.append(self.pd_precision)
            self.pd_all_models_coverage = self.pd_all_models_coverage.append(self.pd_coverage)
            self.pd_all_models_f2s = self.pd_all_models_f2.append(self.pd_f2)
            self.pd_precision.to_csv(self.filename + 'precision.csv', index=False)
            self.pd_coverage.to_csv(self.filename + 'coverage.csv', index=False)
            self.pd_f2.to_csv(self.filename + 'f2.csv', index=False)
            self.pd_all_models_precision.to_csv(filename_all + 'precision.csv', index=False)
            self.pd_all_models_coverage.to_csv(filename_all + 'coverage.csv', index=False)
            self.pd_all_models_f2s.to_csv(filename_all + 'f2.csv', index=False)
        
        if not self.multimodal == []:
            self.final_multimodal = self.multimodal/nb_instance
            self.final_multimodals[nb_model] = self.final_multimodal
        
        if not self.pd_stability.empty:
            self.pd_all_models_stability_features = self.pd_all_models_stability_features.append(self.pd_stability_features)
            self.pd_stability_features.to_csv(self.filename + 'stability_feature.csv', index=False)
            self.pd_all_models_stability = self.pd_all_models_stability.append(self.pd_stability)
            self.pd_stability.to_csv(self.filename + 'stability.csv', index=False)
            self.pd_all_models_stability_features.to_csv(filename_all + 'stability_feature.csv', index=False)
            self.pd_all_models_stability.to_csv(filename_all + 'stability.csv', index=False)

        if not self.pd_msi.empty:
            self.pd_all_models_msi = self.pd_all_models_msi.append(self.pd_msi)
            self.pd_msi.to_csv(self.filename + 'msi.csv', index=False)
            self.pd_all_models_msi.to_csv(filename_all + 'msi.csv', index=False)
            self.pd_all_models_csi = self.pd_all_models_csi.append(self.pd_csi)
            self.pd_csi.to_csv(self.filename + 'csi.csv', index=False)
            self.pd_all_models_csi.to_csv(filename_all + 'csi.csv', index=False)
            self.pd_all_models_vsi = self.pd_all_models_vsi.append(self.pd_vsi)
            self.pd_vsi.to_csv(self.filename + 'vsi.csv', index=False)
            self.pd_all_models_vsi.to_csv(filename_all + 'vsi.csv', index=False)

        if not self.pd_average_distance.empty:
            self.pd_all_models_distance = self.pd_all_models_distance.append(self.pd_average_distance)
            self.pd_average_distance.to_csv(self.filename + 'average_distance.csv', index=False)
            self.pd_all_models_distance.to_csv(filename_all + 'average_distance.csv', index=False)
        
        if not self.pd_lime_ls.empty:
            self.pd_all_models_lime_ls = self.pd_all_models_lime_ls.append(self.pd_lime_ls)
            self.pd_lime_ls.to_csv(self.filename + 'lime_vs_ls.csv', index=False)
            self.pd_all_models_lime_ls.to_csv(filename_all + 'lime_vs_ls.csv', index=False)

        if not self.pd_degrees.empty:
            self.pd_all_models_degrees = self.pd_all_models_degrees.append(self.pd_degrees)
            self.pd_degrees.to_csv(self.filename + "degrees.csv", index=False)
            self.pd_all_models_degrees.to_csv(filename_all + "degrees.csv", index=False)

        if not self.pd_kendall.empty:
            self.pd_all_models_kendall = self.pd_all_models_kendall.append(self.pd_kendall)
            self.pd_kendall.to_csv(self.filename + "kendall.csv", index=False)
            self.pd_all_models_kendall.to_csv(filename_all + "kendall.csv", index=False)
            
        if not self.pd_all_models_top_k.empty:
            self.pd_all_models_top_k.to_csv(filename_all + "mean_top_k.csv", index=False)

        if not self.pd_counterfactual_in_anchor.empty:
            self.pd_all_models_counterfactual_in_anchor = self.pd_all_models_counterfactual_in_anchor.append(self.pd_counterfactual_in_anchor)
            self.pd_counterfactual_in_anchor.to_csv(self.filename + 'counterfactual_in_anchor.csv', index=False)
            self.pd_all_models_counterfactual_in_anchor.to_csv(filename_all + 'counterfactual_in_anchor.csv', index=False)
        
    def store_user_experiments_information_instance(self, recalls, lime=False):
        """
        Store the score of each explanation method computed during the user experiments
        Args: recalls: List of score of each explanation method
        """
        if lime:
            name_filename = 'recall_lime.csv'
        else:
            name_filename = 'recall.csv'
        for recall, interpretability in zip(recalls, self.interpretability_name):
            if self.recall_user_experiments[interpretability] == []:
                self.recall_user_experiments[interpretability] = recall
            else:
                self.recall_user_experiments[interpretability] += recall
        self.pd_recall = self.pd_recall.append(pd.DataFrame([recalls], columns=self.interpretability_name))
        self.pd_recall.to_csv(self.filename + name_filename, index=False)

    def store_user_experiments_information(self, nb_instance, nb_model, filename_all="", lime=False):
        """
        Compute the mean score of each explanation method and store it into an array
        Args: nb_instance: Number of instances for which we generate an explanation for each model
              nb_model: Numerous of the black box model for which we generate explanation (first model employed = 0 , second model employed = 1, etc...)
        """
        os.makedirs(os.path.dirname(self.filename+"/"), exist_ok=True)
        os.makedirs(os.path.dirname(filename_all+"/"), exist_ok=True)
        self.final_recall = []
        for interpretability in self.interpretability_name:
            self.final_recall.append(self.recall_user_experiments[interpretability] / nb_instance)

        for nb, interpretability in enumerate(self.interpretability_name):
            self.final_recalls[nb*self.len_models + nb_model] = self.final_recall[nb]
        self.pd_all_models_recall = self.pd_all_models_recall.append(self.pd_recall)
        if lime:
            name_filename = 'recall_lime.csv'
        else:
            name_filename = 'recall.csv'
        self.pd_recall.to_csv(self.filename + name_filename, index=False)
        self.pd_all_models_recall.to_csv(filename_all + name_filename, index=False)

    def store_stability_information_instance(self, stability_score, stability_features_score):
        self.pd_stability = self.pd_stability.append(pd.DataFrame([stability_score], columns=self.interpretability_name))
        self.pd_stability.to_csv(self.filename + 'stability.csv', index=False)
        self.pd_stability_features = self.pd_stability_features.append(pd.DataFrame([stability_features_score], columns=self.interpretability_name))
        self.pd_stability_features.to_csv(self.filename + 'stability_feature.csv', index=False)

    def store_stability_coefficient_information_instance(self, msi, csi, vsi):
        self.pd_msi = self.pd_msi.append(pd.DataFrame([msi], columns=["APE"]))
        self.pd_csi = self.pd_csi.append(pd.DataFrame([csi], columns=self.interpretability_name))
        self.pd_vsi = self.pd_vsi.append(pd.DataFrame([vsi], columns=self.interpretability_name))
        self.pd_msi.to_csv(self.filename + 'msi.csv', index=False)
        self.pd_csi.to_csv(self.filename + 'csi.csv', index=False)
        self.pd_vsi.to_csv(self.filename + 'vsi.csv', index=False)

    def store_average_distance_instance(self, average_distance, all_average_distance):
        average = np.array([[average_distance, all_average_distance]])
        self.pd_average_distance = self.pd_average_distance.append(pd.DataFrame(average, columns=self.interpretability_name))
        self.pd_average_distance.to_csv(self.filename + 'average_distance.csv', index=False)
    
    def store_lime_vs_local_surrogate(self, k_closest_lime, k_closest_ls, radius):
        k_closest = np.array([[radius, k_closest_lime, k_closest_ls]])
        self.pd_lime_ls = self.pd_lime_ls.append(pd.DataFrame(k_closest, columns=self.interpretability_name))
        self.pd_lime_ls.to_csv(self.filename + 'lime_vs_ls.csv', index=False)
    
    def store_degrees(self, degrees):
        degrees = np.array([degrees])
        self.pd_degrees = self.pd_degrees.append(pd.DataFrame(degrees, columns=self.interpretability_name))
        self.pd_degrees.to_csv(self.filename + "degrees.csv", index=False)
    
    def store_kendall(self, kendall):
        kendall = np.array([kendall])
        self.pd_kendall = self.pd_kendall.append(pd.DataFrame(kendall, columns=self.interpretability_name))
        self.pd_kendall.to_csv(self.filename + "kendall.csv", index=False)

    def store_mean_top_k(self, mean_top):
        mean_top = np.array([mean_top])
        self.pd_all_models_top_k = self.pd_all_models_top_k.append(pd.DataFrame(mean_top, columns=["hit-1", "hit-2", "hit-3", "hit-4", "hit-5"]))

    def store_counterfactual_in_anchor(self, counterfactual_in_anchor):
        counterfactual_in_anchor = np.array([counterfactual_in_anchor])
        self.pd_counterfactual_in_anchor = self.pd_counterfactual_in_anchor.append(pd.DataFrame(counterfactual_in_anchor, columns=self.interpretability_name))
        self.pd_counterfactual_in_anchor.to_csv(self.filename + 'counterfactual_in_anchor.csv', index=False)
