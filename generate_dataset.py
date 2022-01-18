from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from anchors import utils
import numpy as np

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
    
    elif "blood" in dataset_name:
        dataset = utils.load_dataset("blood", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        class_names = ['Donating', 'Not donating']
    
    elif "diabetes" in dataset_name:
        dataset = utils.load_dataset("diabetes", balance=False, discretize=False, dataset_folder="./dataset/")
        x_data, y_data = dataset.train, dataset.labels_train
        class_names = ['Tested Positive', 'Tested Negative']

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

