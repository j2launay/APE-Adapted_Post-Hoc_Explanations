from sklearn.datasets import make_circles, make_moons, make_blobs, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from anchors import utils
import numpy as np
import random

def preparing_dataset(x, y):
    # Split the data inside a test and a train set (70% train and 30% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test

def generate_dataset(dataset_name, multiclass=False):
    # Function used to get dataset depending on the dataset name
    regression = False
    class_names = None
    continuous_features=None 
    categorical_features=[] 
    categorical_values=[]
    categorical_names = []
    transformations = None
    
    if "adult" in dataset_name:
        dataset = utils.load_dataset("adult", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for features in categorical_features:
            tab = [i for i in range(len(dataset.categorical_names[features]))]
            categorical_values.append(tab)
        class_names = ['Less than $50,000', 'More than $50,000']
        categorical_names = dataset.categorical_names
        transformations = dataset.transformations
    
    elif "titanic" in dataset_name:
        dataset = utils.load_dataset("titanic", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for features in categorical_features:
            try:
                tab = list(set(x_data[:,features]))
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        class_names = ['Survive', 'Died']
        categorical_names = dataset.categorical_names
        transformations = dataset.transformations
    
    elif "cancer" in dataset_name:
        dataset = load_breast_cancer()
        x_data, y_data = dataset.data, dataset.target
        class_names = dataset.target_names

    elif "blood" in dataset_name:
        dataset = utils.load_dataset("blood", balance=True, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        
        # Code to balance dataset
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(y_data))
        for label in np.unique(y_data):
            idx = np.random.choice(np.where(y_data == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        x_data = x_data[idxs]
        y_data = y_data[idxs]
        
        class_names = ['Donating', 'Not donating']
    
    elif "diabetes" in dataset_name:
        dataset = utils.load_dataset("diabetes", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        class_names = ['Tested Positive', 'Tested Negative']

    elif "categorical_generate_blobs" in dataset_name:
        x_data, y_data = make_blobs(5000, n_features=8, random_state=0, centers=2, cluster_std=5)
        class_names = ['class ' + str(Y) for Y in range(len(set(y_data)))]
        # we randomly decide the continuous features that will be transform to categorical
        categorical_features = np.random.randint(0,8,(4))
        categorical_features = list(set(categorical_features))
        while len(set(categorical_features)) < 4:
            categorical_features.append(np.random.randint(0,len(x_data[0])))
            categorical_features = list(set(categorical_features))
        for cat_feature in categorical_features:
            # binarize the value of the continuous feature selected to be transform into categorical
            x_data[:,cat_feature] = np.digitize(x_data[:,cat_feature],bins=[np.mean(x_data[:,cat_feature])])
        categorical_values = []
        for i in categorical_features:
            tab = [0,1]
            categorical_values.append(tab)
        categorical_names = {}
        for key, value in zip(categorical_features, categorical_values):
            categorical_names[str(key)] = value
    
    elif "mega_generate_blobs" in dataset_name:
        x_data, y_data = make_blobs(7500, n_features=20, random_state=0, centers=2, cluster_std=5)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
        multiclass = True

    elif "blobs" in dataset_name:
        x_data, y_data = make_blobs(5000, n_features=12, random_state=0, centers=2, cluster_std=5)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
        multiclass = True
    
    elif "blob" in dataset_name:
        x_data, y_data = make_blobs(1000, n_features=2, random_state=0, centers=2, cluster_std=1)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
        multiclass = False
    
    elif "circles" in dataset_name:
        x_data, y_data = make_circles(n_samples=1000, noise=0.05, random_state=0)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
    
    elif 'compas' in dataset_name:
        dataset = utils.load_dataset("compas", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for features in categorical_features:
            try:
                tab = list(set(x_data[:,features]))
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)

        class_names = ['Recidiv', 'Vanish']
        transformations = dataset.transformations

    elif "mortality" in dataset_name:
        dataset = utils.load_dataset("mortality", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        continuous_features, categorical_features = dataset.continuous_features, dataset.categorical_features
        categorical_names, categorical_values = dataset.categorical_names, dataset.categorical_values
        class_names = dataset.class_names
        feature_names = dataset.feature_names

    else:
        # By default the dataset chosen is generate_moons
        dataset_name = "generate_moons"
        x_data, y_data = make_moons(2000, noise=0.2, random_state=0)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
    
    if "generate" in dataset_name or "artificial" in dataset_name:
        if "blobs" in dataset_name:
            alphabet = ["a", "b", "c", "d", "e", "f","g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
            feature_names = alphabet[:len(x_data[0])]
        else:
            feature_names = ["x", "y"]
    else: 
        feature_names = dataset.feature_names

    
    if categorical_features != []:
        continuous_features = [x for x in range(len(x_data[0])) if x not in categorical_features]
    else:
        continuous_features = [x for x in range(len(x_data[0]))]

    return x_data, y_data, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, \
                categorical_names, feature_names, transformations

def modify_dataset(dataset, nb_feature_to_set_0, randomly=False):
    """
    Function that modify the dataset by turning values of data point for a given number of features to 0
    Args: dataset: The dataset that we want to modify
          nb_feature_to_set_0: Number of features for which we want to turn the values of dataset to 0
          randomly: parameter used to replace values in the dataset either by 0 (set to False) or random values (set to True)
    Return: A modified dataset 
    """
    # Create a list of values corresponding to the features that will be replaced to 0
    feature_modified = random.sample(range(0, len(dataset[0])), nb_feature_to_set_0)
    feature_kept = list(set(range(len(dataset[0]))).difference(feature_modified))
    dataset_to_return = dataset.copy()
    
    if randomly:
        # Used in case of replacing values with random values instead of 0
        for feature_modify in feature_modified:
            random_value = np.random.uniform(int(min(dataset[:,feature_modify])), int(max(dataset[:,feature_modify])), len(dataset)).tolist()
            dataset_to_return[:,feature_modify] = random_value 
    else:
        # Modify the dataset to replace values of the computed features by 0
        dataset_to_return[:,feature_modified] = 0
    print("feature kept", feature_kept)
    return dataset_to_return, feature_kept