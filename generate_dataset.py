from types import CodeType
from sklearn.datasets import make_multilabel_classification, make_circles, make_moons, make_blobs, load_boston, fetch_20newsgroups, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from anchors import utils
import numpy as np
import os
from numpy.random import randint
from dataset.mortality import loadnhanes


def preparing_dataset(x, y, dataset_name, plot=False, text=False):
    if plot:
        # Convert data to 2 features in order to be printable
        x = PCA(n_components=2).fit_transform(x)
    #if not text:
        # Store the dataset based on the function from anchors
    #    dataset = utils.load_dataset(dataset_name, balance=False, discretize=False, dataset_folder="./dataset", X=x, y=y, plot=plot)
        
    if not text:
        # Split the data inside a test and a train set (70% train and 30% test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
        vectorizer = None
        return x_train, x_test, y_train, y_test
    else:
        print("Preparing dataset for text data")
        if "newsgroup" in dataset_name:x, _, y, _ = train_test_split(x, y, test_size=0.9, random_state=10)
        x_train_vectorize, x_test_vectorize, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=10)
        vectorizer = CountVectorizer(min_df=1)
        # Si il y a un probleme avec le vectorizer c'est parce qu'il n'a pas les mots du jeu de test lorsqu'il fit
        vectorizer.fit(x_train_vectorize)
        x_train = vectorizer.transform(x_train_vectorize)
        x_test = vectorizer.transform(x_test_vectorize)
        return x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer

def generate_dataset(dataset_name, multiclass=False):
    # Function used to get dataset depending on the dataset name
    regression = False
    class_names = None
    continuous_features=None 
    categorical_features=[] 
    categorical_values=[]
    categorical_names = []
    transformations = None
    if "polarity" in dataset_name:
        path='./dataset/rt-polaritydata'
        x_data = []
        y_data = []
        class_names = ['positive', 'negative']
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(path, f), 'rb'):
                try:
                    line.decode('utf8')
                except:
                    continue
                x_data.append(line.strip())
                y_data.append(l)
    
    elif "newsgroup" in dataset_name:
        newsgroups_train = fetch_20newsgroups(subset='train')
        x_data = newsgroups_train.data
        y_data = newsgroups_train.target
        # making class names shorter
        class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
        multiclass=True
    
    elif "iris" in dataset_name:
        iris = load_iris()
        x_data = iris.data
        y_data = iris.target
        class_names = ['Setosa', 'Versicolour', 'Virginica']
        multiclass=True
    
    elif "adult" in dataset_name:
        dataset = utils.load_dataset("adult", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
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
        for nb, features in enumerate(categorical_features):
            try:
                tab = list(set(x_data[:,features]))
                #tab = [int(x) for x in dataset.categorical_names[features]]
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        class_names = ['Survive', 'Died']
        categorical_names = dataset.categorical_names
        transformations = dataset.transformations
    
    elif "blood" in dataset_name:
        dataset = utils.load_dataset("blood", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        class_names = ['Donating', 'Not donating']
    
    elif "diabete" in dataset_name:
        dataset = utils.load_dataset("diabete", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        class_names = ['Tested Positive', 'Tested Negative']
    
    elif "boston" in dataset_name:
        x_data, y_data = load_boston(True)
        regression = True

    elif "categorical_generate_blobs" in dataset_name:
        x_data, y_data = make_blobs(5000, n_features=8, random_state=0, centers=2, cluster_std=5)
        class_names = ['class ' + str(Y) for Y in range(len(set(y_data)))]
        categorical_features = np.random.randint(0,8,(4))
        categorical_features = list(set(categorical_features))
        while len(set(categorical_features)) < 4:
            categorical_features.append(np.random.randint(0,len(x_data[0])))
            categorical_features = list(set(categorical_features))
        for cat_feature in categorical_features:
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

    elif "plot" in dataset_name:
        x_data, y_data = make_blobs(1000, n_features=2, random_state=0, centers=2, cluster_std=2)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
        multiclass = False
    
    elif "multilabel" in dataset_name:
        x_data, y_data = make_multilabel_classification(n_samples=500, n_classes=3, n_labels=1, allow_unlabeled=False, n_features=2)
        true_y = [0 for i in range(len(y_data))]
        for i, k in enumerate(y_data):
            for j in range (len(y_data[0])):
                if y_data[i][j] == 1:
                    true_y[i] = j
                    break
        y_data = np.array(true_y)
        multiclass=True
    
    elif "artificial" in dataset_name:
        tailles = randint(120, 200, 150)
        ages = randint(20, 40, 150)
        y_data = randint(0, 2, 150)
        instance = []
        for taille, age in zip(tailles, ages):
            instance.append([taille, age])
        x_data = np.array(instance)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
    
    elif "circles" in dataset_name:
        x_data, y_data = make_circles(n_samples=1000, noise=0.05, random_state=0)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y_data)))]
    
    elif 'compas' in dataset_name:
        dataset = utils.load_dataset("compas", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
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
        #categorical_names, categorical_values = dataset.categorical_names, dataset.categorical_values
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
            #feature_names.append("class")
        else:
            feature_names = ["x", "y"]
    else: 
        feature_names = dataset.feature_names

    
    if categorical_features != []:# and not "mortality" in dataset_name:
        categorical_feature_names = [feature_names[index] for index in categorical_features]
        continuous_features = [x for x in range(len(x_data[0])) if x not in categorical_features]
        continuous_feature_names = [feature_names[index] for index in continuous_features]
        enc = OneHotEncoder(handle_unknown='ignore')
        categorical_x = x_data[:,categorical_features]
        enc.fit(categorical_x)
        x_transform = enc.transform(categorical_x).toarray()
        features = [i for i in range(len(x_data[0]))]
        x_data = np.append(x_transform, x_data[:,continuous_features], axis=1)
        continuous_features = range(len(x_transform[0]), len(x_data[0]))
        categorical_features = range(len(x_transform[0]))
        categorical_values = []
        for i in categorical_features:
            categorical_values.append([0, 1])
        feature_names = np.append(enc.get_feature_names(categorical_feature_names), continuous_feature_names)
        categorical_names = {}
        for key, values in zip(categorical_features, categorical_values):
            categorical_names[key] = [str(value) for value in values]
    #elif 'mortality' not in dataset_name:
    else:
        continuous_features = [x for x in range(len(x_data[0]))]
    #print(categorical_names)
    return x_data, y_data, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, \
                categorical_names, feature_names, transformations

#generate_dataset("mortality")