import pandas as pd
import os

class store_experimental_informations(object):
    """
    Class to store the experimental results of accuracy, coverage and F1 score for graph representation
    """
    def __init__(self, columns_name_file1, columns_name_file2=None, columns_name_file3=None, columns_multimodal=None):
        """
        Initialize all the variable that will be used to store experimental results
        Args: columns_name_file1: List of the name of the explanation methods used to explain each model
              columns_name_file2 and columns_name_file3 are the columns name used in case there are multiple files with different columns name.
              columns_multimodal: Corresponds to the columns name of the complex file containing supplementary informations such as the result of the separability test 
        """
        columns_name_file2 = columns_name_file1 if columns_name_file2 is None else columns_name_file2
        columns_name_file3 = columns_name_file1 if columns_name_file3 is None else columns_name_file3
        columns_name_file4 = columns_name_file1

        self.multimodal_columns = ["LS", 'LS roc', 'LSE ini', 'LSe ini auc', 'LSe', 'LSe auc', 'LIME', 'LIME roc', 'Anchors', 'DT', 
                                    'APEa', 'APEt', "Multimodal", "radius", "fr pvalue", "cf pvalue", "separability", "fr fold",
                                    "cf fold", "bb", "dataset"] if columns_multimodal == None else columns_multimodal
        self.columns_name_file1 = columns_name_file1
        self.columns_name_file2 = columns_name_file2
        self.columns_name_file3 = columns_name_file3
        self.columns_name_file4 = columns_name_file4

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
        Store experimental results inside dictionary 
        Args: results1: list of result for each explanation method on a single instance
              filename1: Name of the csv file that will be stored in the results folder
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

    def store_experiments_information(self, filename1, filename2=None, filename3=None, 
            filename4=None, filename_multimodal=None, filename_all="", multimodal_filename="multimodal.csv"):
        """ 
        Compute the mean results per model 
        Args: filename1: Name of the csv file that will be stored in the results folder
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
        