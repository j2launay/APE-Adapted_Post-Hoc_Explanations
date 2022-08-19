# APE: Adapted Post-Hoc Explanations
This repository contains the code presented in the paper ``When Should We Use Linear Explanations?'', as well as some experiments and results made for the publication.

Python 3.7 version required
In order to install [Libfolding](https://github.com/asiffer/libfolding) the library used to adapt the explanation based on the distributino of data, you must install [Armadillo](http://arma.sourceforge.net/download.html). Therafter install libfolding following indications from the github.  

* Code to launch the experiments for accuracy, coverage, F2 and proportion of multimodal are in tabular_experiments.py (Table 3 in Section 4.2 *adherence evaluation* as well as Figure 4 and 5 from section 4.3) Results from Figure 5 a and b are obtained from tabular_experiments.py and computed over the file multimodal.csv containing the results of the Thornton's linear separability test and the unimodal and multimodal test for each instance
* Code to launch the experiments for the closest real instances between counterfactual found with Growing Fields and Growing Sphere (Figure 6 section 4.5) is k_closest_experiments.py

All the results computed are then available in the folder results/ following by the dataset name, the black box employed and the threshold for the interpretability method.

ape_tabular.py is the main class, it is used as the code to explain a particular instance with any black box model and to return either a rule based explanation or a linear explanation along one or multiple counterfactual.
ape_experiments_functions.py contain codes to compute the average distance to the k closest counterfactual from Figure 6 in Section 4.5 and the average adherence and fidelity of each explanation module.

run: pip install -r requirements.txt in your shell in order to install all the packages that are necessary to launch the experiments.

## Datasets and Classifiers
This section provides useful information to reproduce the experimental results from the paper:
* Each dataset can be download from the link listed in Table 1;
* The parameters required to replicate the scikit-learn artificial datasets are provided in Table 2.

### Dataset links:
<table>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Name</strong></td>
        <td><strong>Link</strong></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Adult</td>
        <td>https://archive.ics.uci.edu/ml/datasets/adult</td>
    </tr>
    <tr>
        <td>Blood</td>
        <td>https://www.openml.org/d/1464</td>
    </tr>
    <tr>
        <td>Diabete</td>
        <td>https://www.openml.org/d/37</td>
    </tr>
    <tr>
        <td>Compas</td>
        <td>https://github.com/propublica/compas-analysis/</td>
    </tr>
    <tr>
        <td>Mortality</td>
        <td>https://github.com/suinleelab/treeexplainer-study/tree/master/notebooks/mortality</td>
    </tr>
    <tr>
        <td>Titanic</td>
        <td>https://www.kaggle.com/c/titanic/overview</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
</table>
Table 1: Dataset links.  <br/><br/>

### Parameters set to generate the artificial datasets
Cat Blobs dataset is generated through the make$\_$blobs function (Blob, Blobs or M Blobs from Table 2) where 4 features have been randomly discretized into binary variables depending on either the feature value is (1) superior or (0) inferior to the mean feature value of the dataset.  
Table 2 summarizes for each dataset -- i.e: synthetic and real -- the number of instances as well as the number of categorical and numerical attributes.

<table>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Name</strong></td>
        <td><strong>Parameters</strong></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Blob</td>
        <td>make_blobs(1000, n_features=2, random_state=0, centers=2, cluster_std=1)</td>
    </tr>
    <tr>
        <td>Blobs</td>
        <td>make_blobs(n_samples=5000, n_features=12, random_state=0, centers=2, cluster_std=5)</td>
    </tr>
    <tr>
        <td>Cancer</td>
        <td>load_breast_cancer()</td>
    </tr>
    <tr>
        <td>Circles</td>
        <td>make_circles(n_samples=1000, noise=0.05, random_state=0)</td>
    </tr>
    <tr>
        <td>M Blobs</td>
        <td>make_blobs(7500, n_features=20, random_state=0, centers=2, cluster_std=5)</td>
    </tr>
    <tr>
        <td>Moons</td>
        <td>make_moons(n_samples=2000, noise=0.2, random_state=0)</td>
    </tr>
</table>
Table 2: Parameters set to generate the artificial datasets  <br/><br/>

### Parameters to set the classifiers and their accuracy by black box
Regarding the classifiers, 6 black-box models from scikit-learn have been used. Table 3 shows the hyperparameters set for learning each classifier. We split each dataset into training (70%) and testing (30%) sets.   
<table>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Name</strong></td>
        <td><strong>Parameters</strong></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>GB</td>
        <td>GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,random_state=1)</td>
    </tr>
    <tr>
        <td>MLP</td>
        <td>MLPClassifier(random_state=1)</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>RandomForestClassifier(n_estimators=20, random_state=1)</td>
    </tr>
    <tr>
        <td>VOT</td>
        <td>VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('svm', svm.SVC(probability=True))], voting="soft")</td>
    </tr>
    <tr>
        <td>RC</td>
        <td>RidgeClassifier(random\_state=1)</td>
    </tr>
    <tr>
        <td>SVM</td>
        <td>SVC(probability=True, random\_state=1)</td>
    </tr>
</table>


## Time comparison
* Code to compare the runtime of Growing Spheres and Growing Fields is computed in time_closest.py

The following Table 3 presents runtime performances for the closest counterfactual search of both algorithms: Growing Spheres and Growing Fields:

<table>
    <tr>
        <td></td>
        <td colspan="2"><center>GB</center></td>
        <td colspan="2"><center>MLP</center></td>
        <td colspan="2"><center>RF</center></td>
        <td colspan="2"><center>RC</center></td>
        <td colspan="2"><center>VC</center></td>
    </tr>
    <tr>
        <td></td>
        <td>GS</td>
        <td>GF</td>
        <td>GS</td>
        <td>GF</td>
        <td>GS</td>
        <td>GF</td>
        <td>GS</td>
        <td>GF</td>
        <td>GS</td>
        <td>GF</td>
    </tr>
    <tr>
        <td>Blood</td>
        <td>31.8</td>
        <td><strong>0.14</strong></td>
        <td>212</td>
        <td><strong>0.26</strong></td>
        <td>96.4</td>
        <td><strong>0.70</strong></td>
        <td>371</td>
        <td><strong>0.11</strong></td>
        <td>725</td>
        <td><strong>2.09</strong></td>
    </tr>
    <tr>
        <td>Blob</td>
        <td>6.71</td>
        <td><strong>0.12</strong></td>
        <td>15.6</td>
        <td><strong>0.18</strong></td>
        <td>12.7</td>
        <td><strong>0.47</strong></td>
        <td>14.6</td>
        <td><strong>0.08</strong></td>
        <td>38.2</td>
        <td><strong>1.50</strong></td>
    </tr>
    <tr>
        <td>Blobs</td>
        <td>82.9</td>
        <td><strong>0.42</strong></td>
        <td>158</td>
        <td><strong>0.53</strong></td>
        <td>153</td>
        <td><strong>0.86</strong></td>
        <td>162</td>
        <td><strong>0.35</strong></td>
        <td>408</td>
        <td><strong>2.47</strong></td>
    </tr>
    <tr>
        <td>Circles</td>
        <td>1.53</td>
        <td><strong>0.19</strong></td>
        <td>2.05</td>
        <td><strong>0.30</strong></td>
        <td>2.21</td>
        <td><strong>0.65</strong></td>
        <td>10.2</td>
        <td><strong>0.12</strong></td>
        <td>26.8</td>
        <td><strong>3.03</strong></td>
    </tr>
    <tr>
        <td>Diabetes</td>
        <td>4.36</td>
        <td><strong>0.21</strong></td>
        <td>101</td>
        <td><strong>0.28</strong></td>
        <td>45.7</td>
        <td><strong>0.58</strong></td>
        <td>23.4</td>
        <td><strong>0.14</strong></td>
        <td>42.8</td>
        <td><strong>2.14</strong></td>
    </tr>
    <tr>
        <td>M Blobs</td>
        <td>237</td>
        <td><strong>1.00</strong></td>
        <td>260</td>
        <td><strong>0.84</strong></td>
        <td>315</td>
        <td><strong>1.69</strong></td>
        <td>241</td>
        <td><strong>0.62</strong></td>
        <td>573</td>
        <td><strong>2.82</strong></td>
    </tr>
    <tr>
        <td>Moons</td>
        <td>1.69</td>
        <td><strong>0.15</strong></td>
        <td>4.24</td>
        <td><strong>0.20</strong></td>
        <td>3.41</td>
        <td><strong>0.47</strong></td>
        <td>5.24</td>
        <td><strong>0.09</strong></td>
        <td>14.0</td>
        <td><strong>1.56</strong></td>
    </tr>
</table>
Table 3: Average time (second) over 100 instances per black box and dataset to find the closest counterfactual by Growing Spheres (GS) and Growing Fields (GF).<br/><br/>

As mentioned in Section 4.5 of the paper, Growing Fields algorithm relies on the dataset distribution to sample artificial instances and generate the closest artificial counterfactual. Thus, we prove in this Table with numerical results that employing the mean and standard deviation of each feature to samples instances allows Growing Fields to increase the values of specific features with potentially high values to rise with a different speed -- i.e: salary compared to age.-- We notice that on average over 100 instances on 7 datasets and 5 black boxes, Growing Fields discover with a mean increase of 2 orders of magnitude the closest counterfactual faster than Growing Spheres. We also remark that generating a counterfactual with Growing Fields takes between 0.08 and 3.03 seconds, thus with few runtime variation while for Growing Sphere it varies between 1.53 and 725 seconds. Hence, Growing Fields allows to obtain faster more reliable counterfactual explanations than Growing Spheres.
