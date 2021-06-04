# APE: Adapted Post-Hoc Explanations
This repository contains the code presented in the paper APE: Adapted Post-Hoc Explanations, as well as some experiments and results made for the publication.

Python 3.7 version required
In order to install [Libfolding](https://github.com/asiffer/libfolding) the library used to adapt the explanation based on the distributino of data, you must install [Armadillo](http://arma.sourceforge.net/download.html). Therafter install libfolding following indications from the github.  

Code to launch the experiments for precision, coverage, F1 and proportion of multimodal are in tabular_experiments.py
Code to launch the simulated users experiments are in tabular_user_experiments.py and tabular_user_experiments_lime.py corresponds to the function used for Lime vs Local Surrogate experiments.

Ape_tabular is the main class, it is used as the code to explain a particular instance with any black box model and to return either a rule based explanation or a linear explanation along one or multiple counterfactuals.

run: pip install -r requirements.txt in your shell in order to install all the packages that are necessary to launch the experiments.
