import numpy as np
import csv

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
        
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'grey', 'purple']
    color= []
    for nb, model in enumerate(models):
        color.append(colors[nb])
    return color, bars, y_pos

class store_experimental_informations(object):
    """
    Class to store the experimental results of precision, coverage and F1 score for graph representation
    """
    def __init__(self, len_models, len_interpretability_name, interpretability_name):
        """
        Initialize all the variable that will be used to store experimental results
        Args: len_models: Number of black box models that we are explaining during experiments
              len_interpretability_name: Number of explanation methods used to explain each model
              interpretability_name: List of the name of the explanation methods used to explain each model 
        """
        self.interpretability_name = interpretability_name
        self.len_interpretability_name, self.len_models, = len_interpretability_name, len_models
        size_results = len_models*len_interpretability_name
        self.final_precisions, self.final_coverages, self.final_f1s = np.zeros(size_results), np.zeros(size_results), np.zeros(size_results)
        self.final_recalls = np.zeros(size_results)
        self.final_multimodals = np.zeros(len_models)

    def initialize_per_models(self):
        self.precision = {}
        self.coverage = {}
        self.f1 = {}
        self.recall_user_experiments = {}
        self.multimodal = []
        for interpretability in self.interpretability_name:
            self.precision[interpretability] = []
            self.coverage[interpretability] = []
            self.f1[interpretability] = []
            self.recall_user_experiments[interpretability] = []

    def store_experiments_information_instance(self, precisions, coverages, f1s, multimodal=None, filename=None):
        """
        Store precisions, coverages, f1s and multimodal results inside dictionary 
        Args: precisions: list of precision result for each explanation method on a single instance
              coverages: list of coverage result for each explanation method on a single instance
              f1s: list of f1 score for each explanation method on a single instance
              multimodal: 1 if APE selected a multimodal distribution, otherwise 0
        """
        if filename != None:
            with open(filename + "/tab_resume.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                test = ['']*4
                text = []
                metrics = ["precision", "coverage", "f1s"]
                for metric in metrics:
                    text.append(metric)
                    for element in test:
                        text.append(element)
                    text.append('')
                print(text)
                writer.writerow(text)
                writer.writerow([precisions, '', coverages, '', f1s])
                writer.writerow([])
                #writer.writerow(['anchors precision'])
                #writer.writerow([self.output_csv(writer, precision, x)])
        
        for precision, coverage, f1, interpretability in zip(precisions, coverages, f1s, self.interpretability_name):
            if self.precision[interpretability] == []:
                self.precision[interpretability] = precision
                self.coverage[interpretability] = coverage
                self.f1[interpretability] = f1
            else:
                self.precision[interpretability] += precision
                self.coverage[interpretability] += coverage
                self.f1[interpretability] += f1
        if multimodal is not None:
            if self.multimodal == []:
                self.multimodal = multimodal
            else:
                self.multimodal += multimodal

    def store_experiments_information(self, nb_instance, nb_model):
        """ 
        Compute the mean coverage, precision and f1 per model 
        Args: nb_instance: Number of instance for which we generate explanation for each model
              nb_model: Numerous of the black box model for which we generate explanation (first model employed = 0 , second model employed = 1, etc...)
        """
        self.final_precision = []
        self.final_coverage = []
        self.final_f1 = []
        for interpretability in self.interpretability_name:
            # For each explanation method we compute the mean precision, coverage and f1
            self.final_precision.append(self.precision[interpretability] / nb_instance)
            self.final_coverage.append(self.coverage[interpretability] / nb_instance)
            self.final_f1.append(self.f1[interpretability] / nb_instance)
        for nb, interpretability in enumerate(self.interpretability_name):
            # Store in arrays the mean precision, coverage and f1
            self.final_precisions[nb*self.len_models + nb_model] = self.final_precision[nb]
            self.final_coverages[nb*self.len_models + nb_model] = self.final_coverage[nb]
            self.final_f1s[nb*self.len_models + nb_model] = self.final_f1[nb]
        
        if self.multimodal != []:
            self.final_multimodal = self.multimodal/nb_instance
            self.final_multimodals[nb_model] = self.final_multimodal
        
    def store_user_experiments_information_instance(self, recalls):
        """
        Store the score of each explanation method computed during the user experiments
        Args: recalls: List of score of each explanation method
        """
        for recall, interpretability in zip(recalls, self.interpretability_name):
            if self.recall_user_experiments[interpretability] == []:
                self.recall_user_experiments[interpretability] = recall
            else:
                self.recall_user_experiments[interpretability] += recall
        print(self.recall_user_experiments)
    
    def store_user_experiments_information(self, nb_instance, nb_model):
        """
        Compute the mean score of each explanation method and store it into an array
        Args: nb_instance: Number of instances for which we generate an explanation for each model
              nb_model: Numerous of the black box model for which we generate explanation (first model employed = 0 , second model employed = 1, etc...)
        """
        self.final_recall = []
        for interpretability in self.interpretability_name:
            self.final_recall.append(self.recall_user_experiments[interpretability] / nb_instance)

        for nb, interpretability in enumerate(self.interpretability_name):
            self.final_recalls[nb*self.len_models + nb_model] = self.final_recall[nb]

def write_results(precisions, coverages, f1s, filename, interpretability_name, metrics_name):
    with open(filename + "/tab_resume.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        test = ['']*len(precisions)
        text = []
        inter = []
        for metric in metrics_name:
            text.append(metric)
            for element in test:
                text.append(element)
            for interpretability in interpretability_name:
                inter.append(interpretability)
            inter.append("")
        writer.writerow(text)
        writer.writerow(inter)
        results = [a for a in precisions]
        results.append("")
        for b in coverages:
            results.append(b)
        results.append("")
        for c in f1s:
            results.append(c)
        results.append("")
        writer.writerow(results)
        writer.writerow([])

with open("./tab_resume.csv", 'w', newline='') as f:
    precisions = [0.9, 0.95, 0.92, 0.98]
    coverages = [0.5, 0.54, 0.67, 0.76]
    f1s = [0.23, 1, 0.95, 0.12]
    writer = csv.writer(f)
    test = ['']*len(precisions)
    text = []
    inter = []
    interpretability = ["lime", "anchors", 'random', 'APE']
    metrics = ["precision", "coverage", "f1s"]
    for metric in metrics:
        text.append(metric)
        for element in test:
            text.append(element)
        for interpret in interpretability:
            inter.append(interpret)
        inter.append("")
    writer.writerow(text)
    writer.writerow(inter)
    results = [a for a in precisions]
    results.append("")
    for b in coverages:
        results.append(b)
    results.append("")
    for c in f1s:
        results.append(c)
    results.append("")
    writer.writerow(results)
    writer.writerow([])
