from sklearn.datasets import make_multilabel_classification, make_circles, make_moons, make_blobs, load_boston, fetch_20newsgroups, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from anchors import utils
import numpy as np
import os
from numpy.random import randint
import pandas as pd

def preparing_dataset(x, y, dataset_name, model, plot=False, text=False):
    if plot:
        # Convert data to 2 features in order to be printable
        x = PCA(n_components=2).fit_transform(x)
    if not text:
        # Store the dataset based on the function from anchors
        dataset = utils.load_dataset(dataset_name, balance=False, discretize=False, dataset_folder="./dataset", X=x, y=y, plot=plot)
        
    if not text:
        # Split the data inside a test and a train set (50% each)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        vectorizer = None
        return dataset, model, x_train, x_test, y_train, y_test
    else:
        print("Preparing dataset for text data")
        if "newsgroup" in dataset_name:x, _, y, _ = train_test_split(x, y, test_size=0.9)
        x_train_vectorize, x_test_vectorize, y_train, y_test = train_test_split(x, y, test_size=0.5)
        vectorizer = CountVectorizer(min_df=1)
        # Si il y a un probleme avec le vectorizer c'est parce qu'il n'a pas les mots du jeu de test lorsqu'il fit
        vectorizer.fit(x_train_vectorize)
        x_train = vectorizer.transform(x_train_vectorize)
        x_test = vectorizer.transform(x_test_vectorize)
        return None, model, x_train, x_test, y_train, y_test, x_train_vectorize, x_test_vectorize, vectorizer

def generate_dataset(dataset_name, multiclass=False):
    # Function used to get dataset depending on the dataset name
    regression = False
    class_names = None
    continuous_features=None 
    categorical_features=[] 
    categorical_values=[]
    categorical_names = []
    if "polarity" in dataset_name:
        path='./dataset/rt-polaritydata'
        X = []
        y = []
        class_names = ['positive', 'negative']
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(path, f), 'rb'):
                try:
                    line.decode('utf8')
                except:
                    continue
                X.append(line.strip())
                y.append(l)
    
    elif "newsgroup" in dataset_name:
        newsgroups_train = fetch_20newsgroups(subset='train')
        X = newsgroups_train.data
        y = newsgroups_train.target
        # making class names shorter
        class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
        multiclass=True
    
    elif "iris" in dataset_name:
        iris = load_iris()
        X = iris.data
        y = iris.target
        class_names = ['Setosa', 'Versicolour', 'Virginica']
        multiclass=True
    
    elif "adult" in dataset_name:
        dataset = utils.load_dataset("adult", balance=False, discretize=False, dataset_folder="./dataset/")
        X, y = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
            tab = [i for i in range(len(dataset.categorical_names[features]))]
            categorical_values.append(tab)
        class_names = ['Less than $50,000', 'More than $50,000']
        categorical_names = dataset.categorical_names
    
    elif "titanic" in dataset_name:
        dataset = utils.load_dataset("titanic", balance=False, discretize=False, dataset_folder="./dataset/")
        X, y = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
            try:
                tab = [int(x) for x in dataset.categorical_names[features]]
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        class_names = ['Survive', 'Died']
        categorical_names = dataset.categorical_names
    
    elif "blood" in dataset_name:
        dataset = utils.load_dataset("blood", balance=False, discretize=False, dataset_folder="./dataset/")
        X, y = dataset.train, dataset.labels_train
        class_names = ['Donating', 'Not donating']
    
    elif "diabete" in dataset_name:
        dataset = utils.load_dataset("diabete", balance=False, discretize=False, dataset_folder="./dataset/")
        X, y = dataset.train, dataset.labels_train
        class_names = ['Tested Positive', 'Tested Negative']
    
    elif "boston" in dataset_name:
        X, y = load_boston(True)
        regression = True
    
    elif "blobs" in dataset_name:
        X, y = make_blobs(5000, n_features=12, random_state=0, centers=2)#, cluster_std=10)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y)))]
        multiclass=True
    
    elif "blob" in dataset_name:
        X, y = make_blobs(1000, n_features=2, random_state=0, centers=2)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y)))]
        multiclass=True
    
    elif "multilabel" in dataset_name:
        X, y = make_multilabel_classification(n_samples=500, n_classes=3, n_labels=1, allow_unlabeled=False, n_features=2)
        true_y = [0 for i in range(len(y))]
        for i, k in enumerate(y):
            for j in range (len(y[0])):
                if y[i][j] == 1:
                    true_y[i] = j
                    break
        y = np.array(true_y)
        multiclass=True
    
    elif "artificial" in dataset_name:
        tailles = randint(120, 200, 150)
        ages = randint(20, 40, 150)
        y = randint(0, 2, 150)
        instance = []
        for taille, age in zip(tailles, ages):
            instance.append([taille, age])
        X = np.array(instance) 
        class_names = ['class ' + str(Y)  for Y in range(len(set(y)))]
    
    elif "circles" in dataset_name:
        X, y = make_circles(n_samples=1000, noise=0.05)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y)))]
    
    elif 'compas' in dataset_name:
        dataset = utils.load_dataset("compas", balance=False, discretize=False, dataset_folder="./dataset/")
        X, y = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
            try:
                tab = [int(x) for x in dataset.categorical_names[features]]
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        """
        data = pd.read_csv("./dataset/compas/compas_numpy.csv")

        #data[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 
        #                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        filter_data = data[data['days_b_screening_arrest'].notnull()]
        filter_data[filter_data['days_b_screening_arrest'] <= 30] 
        filter_data[filter_data['days_b_screening_arrest'].astype(int) >= -30]
        filter_data[filter_data['is_recid'] != -1] 
        filter_data[filter_data['c_charge_degree'] != "O"]
        #filter_data[filter_data['score_text'] != "N/A"]

        y = data['two_year_recid'].tolist()
        X = data.drop(['two_year_recid'], axis=1).values
        """
        class_names = ['Recidiv', 'Vanish']

    else:
        # By default the dataset chosen is generate_moons
        dataset_name = "generate_moons"
        X, y = make_moons(1000, noise=0.6)
        class_names = ['class ' + str(Y)  for Y in range(len(set(y)))]
    features = [i for i in range(len(X[0]))]
    continuous_features = [x for x in features if x not in categorical_features]
    return X, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names

#generate_dataset("titanic")