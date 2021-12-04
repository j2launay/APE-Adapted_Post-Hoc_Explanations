import pandas as pd
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
    def __init__(self, len_models, len_interpretability_name, columns_name_file1, nb_models, columns_name_file2=None, columns_name_file3=None, columns_multimodal=None):
        """
        Initialize all the variable that will be used to store experimental results
        Args: len_models: Number of black box models that we are explaining during experiments
              len_interpretability_name: Number of explanation methods used to explain each model
              interpretability_name: List of the name of the explanation methods used to explain each model 
        """
        columns_name_file2 = columns_name_file1 if columns_name_file2 is None else columns_name_file2
        columns_name_file3 = columns_name_file1 if columns_name_file3 is None else columns_name_file3
        columns_name_file4 = columns_name_file1

        self.multimodal_columns = ["LS", "LSe log", "LSe lin", "Anchors", "APE", "DT", "Multimodal",
                                    "radius", "fr pvalue", "cf pvalue", "separability", "fr fold",
                                    "cf fold", "SI bon", "fold bon", "ape bon", "bb"] if columns_multimodal == None else columns_multimodal
        self.columns_name_file1 = columns_name_file1
        self.columns_name_file2 = columns_name_file2
        self.columns_name_file3 = columns_name_file3
        self.columns_name_file4 = columns_name_file4
        self.len_interpretability_name, self.len_models, = len_interpretability_name, len_models

        self.nb_models = nb_models - 1
        self.pd_all_models_results1 = pd.DataFrame(columns=columns_name_file1)
        self.pd_all_models_results2 = pd.DataFrame(columns=columns_name_file2)
        self.pd_all_models_results3 = pd.DataFrame(columns=columns_name_file3)
        self.pd_all_models_results4 = pd.DataFrame(columns=columns_name_file4)
        self.pd_all_models_multimodal = pd.DataFrame(columns=self.multimodal_columns)

    def initialize_per_models(self, filename):
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        self.pd_results1 = pd.DataFrame(columns=self.columns_name_file1)
        self.pd_results2 = pd.DataFrame(columns=self.columns_name_file2)
        self.pd_results3 = pd.DataFrame(columns=self.columns_name_file3)
        self.pd_results4 = pd.DataFrame(columns=self.columns_name_file4)
        
        self.pd_multimodal = pd.DataFrame(columns=self.multimodal_columns)
    
    def store_experiments_information_instance(self, results1, filename1, results2=None, filename2=None, results3=None,  
                        filename3=None, results4=None, filename4=None, multimodal=None, multimodal_filename="multimodal.csv"):
        """
        Store precisions, coverages, f2s and multimodal results inside dictionary 
        Args: precisions: list of precision result for each explanation method on a single instance
              coverages: list of coverage result for each explanation method on a single instance
              f2s: list of f2 score for each explanation method on a single instance
              multimodal: 1 if APE selected a multimodal distribution, otherwise 0
        """
        
        self.pd_results1 = self.pd_results1.append(pd.DataFrame([results1], columns=self.columns_name_file1))
        self.pd_results1.to_csv(self.filename + filename1, index=False)
        
        if results2 is not None:
            self.pd_results2 = self.pd_results2.append(pd.DataFrame([results2], columns=self.columns_name_file2))
            self.pd_results2.to_csv(self.filename + filename2, index=False)

        if results3 is not None:
            self.pd_results3 = self.pd_results3.append(pd.DataFrame([results3], columns=self.columns_name_file3))
            self.pd_results3.to_csv(self.filename + filename3, index=False)
        
        if results4 is not None:
            self.pd_results4 = self.pd_results4.append(pd.DataFrame([results4], columns=self.columns_name_file4))
            self.pd_results4.to_csv(self.filename + filename4, index=False)

        if multimodal is not None:
            self.pd_multimodal = self.pd_multimodal.append(pd.DataFrame([multimodal], columns=self.multimodal_columns))
            self.pd_multimodal.to_csv(self.filename + multimodal_filename, index=False)

    def store_experiments_information(self, nb_instance, nb_model, filename1, filename2=None, filename3=None, 
            filename4=None, filename_multimodal=None, filename_all="", multimodal_filename="multimodal.csv"):
        """ 
        Compute the mean coverage, precision and f2 per model 
        Args: nb_instance: Number of instance for which we generate explanation for each model
              nb_model: Numerous of the black box model for which we generate explanation (first model employed = 0 , second model employed = 1, etc...)
        """
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        os.makedirs(os.path.dirname(filename_all), exist_ok=True)

        self.pd_all_models_results1 = self.pd_all_models_results1.append(self.pd_results1)
        self.pd_results1.to_csv(self.filename + filename1, index=False)
        self.pd_all_models_results1.to_csv(filename_all + filename1, index=False)

        if filename2 is not None:
            self.pd_all_models_results2 = self.pd_all_models_results2.append(self.pd_results2)
            self.pd_results2.to_csv(self.filename + filename2, index=False)
            self.pd_all_models_results2.to_csv(filename_all + filename2, index=False)
        
        if filename3 is not None:
            self.pd_all_models_results3 = self.pd_all_models_results3.append(self.pd_results3)
            self.pd_results3.to_csv(self.filename + filename3, index=False)
            self.pd_all_models_results3.to_csv(filename_all + filename3, index=False)
        
        if filename4 is not None:
            self.pd_all_models_results4 = self.pd_all_models_results4.append(self.pd_results4)
            self.pd_results4.to_csv(self.filename + filename4, index=False)
            self.pd_all_models_results4.to_csv(filename_all + filename4, index=False)

        if filename_multimodal is not None:
            self.pd_all_models_multimodal = self.pd_all_models_multimodal.append(self.pd_multimodal)
            self.pd_multimodal.to_csv(self.filename + multimodal_filename, index=False)
            self.pd_all_models_multimodal.to_csv(filename_all + multimodal_filename, index=False)
        
        