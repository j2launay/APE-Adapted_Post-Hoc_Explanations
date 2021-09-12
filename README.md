# APE: Adapted Post-Hoc Explanations
This repository contains the code presented in the paper APE: Adapted Post-Hoc Explanations, as well as some experiments and results made for the publication.

Python 3.7 version required
In order to install [Libfolding](https://github.com/asiffer/libfolding) the library used to adapt the explanation based on the distributino of data, you must install [Armadillo](http://arma.sourceforge.net/download.html). Therafter install libfolding following indications from the github.  

Code to launch the experiments for precision, coverage, F1 and proportion of multimodal are in tabular_experiments.py (Table 2 and 11)
Code to launch the experiments for the closest real instances between counterfactual found with Growing Fields and Growing Sphere (Table 3) is k_closest_experiments.py
Code to launch the simulated users experiments are in tabular_user_experiments.py and tabular_user_experiments_lime.py corresponds to the function used for Lime vs Local Surrogate experiments (Table 4.)
Code to launch the experiments to compare the precision of LIME and Local Surrogate over generated instances depending on the radius of the field (Figure 2 to 6) is in lime_vs_local_surrogate.py and the code to generate the graph is in generate_boxplot.py
Code to launch the experiments for precision and coverage in supplementary materials (Table 5 and 6) is in tabular_experiments_supp_mat.py
Results from Table 10 about the multimodal results is obtain from the code in tabular_experiments.py
Code to launch the experiments for average precision of Local Surrogate employing the Growing Spheres algorithm versus  Growing Fields (Table 12) is obtain in ls_gs_gf_precision.py
All the results computed are then available in the folder results/ following by the dataset name, the black box employed and the threshold for the interpretabbility method.

Ape_tabular is the main class, it is used as the code to explain a particular instance with any black box model and to return either a rule based explanation or a linear explanation along one or multiple counterfactuals.
ape_tabular_experiments contain codes to computes the gold standard features, the precision of the 4 Local Surrogate presented in Table 5 and 6.

run: pip install -r requirements.txt in your shell in order to install all the packages that are necessary to launch the experiments.
