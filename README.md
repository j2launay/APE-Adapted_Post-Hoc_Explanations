# APE: Adapted Post-Hoc Explanations
This repository contains the code presented in the paper APE: Adapted Post-Hoc Explanations, as well as some experiments and results made for the publication.

Python 3.7 version required
In order to install [Libfolding](https://github.com/asiffer/libfolding) the library used to adapt the explanation based on the distributino of data, you must install [Armadillo](http://arma.sourceforge.net/download.html). Therafter install libfolding following indications from the github.  

* Code to launch the experiments for accuracy, coverage, F2 and proportion of multimodal are in tabular_experiments.py (Table 1 and 2 section 4.2 and 4.3)
* Code to launch the experiments for the closest real instances between counterfactual found with Growing Fields and Growing Sphere (Table 3 section 4.4) is k_closest_experiments.py
* Code to launch the experiments for precision and coverage in supplementary materials (Table 8 Appendix B) is in tabular_experiments_supp_mat.py
* Results from Table 9 (Appendix C) about the multimodal results is obtain from the code in tabular_experiments.py
* Results from Table 10 and 11 (Appendix D)  are obtained from tabular_experiments.py and computed over the file multimodal.csv containing the results of the Thornton's linear separability test and the unimodal and multimodal test for each instance
* Code to launch the experiments for average precision of Local Surrogate employing the Growing Spheres algorithm versus  Growing Fields (Table 12 Appendix E) is obtain in ls_gs_gf_precision.py
* Code to compare the runtime from Table 13 Appendix F is computed in time_closest.py

### Additionnal experiments not added in IJCAI
* Code to launch the simulated users experiments are in tabular_user_experiments.py and tabular_user_experiments_lime.py corresponds to the function used for Lime vs Local Surrogate experiments
* Code to launch the experiments to compare the precision of LIME and Local Surrogate over generated instances depending on the radius of the field is in lime_vs_local_surrogate.py and the code to generate the graph is in generate_boxplot.py

All the results computed are then available in the folder results/ following by the dataset name, the black box employed and the threshold for the interpretability method.

Ape_tabular is the main class, it is used as the code to explain a particular instance with any black box model and to return either a rule based explanation or a linear explanation along one or multiple counterfactuals.
ape_experiments_functions.py contain codes to compute the average distance to the k closest counterfactual from Table 3 in Section 4.4 and the average accuracy of the 4 Local Surrogate presented in Table 8.

run: pip install -r requirements.txt in your shell in order to install all the packages that are necessary to launch the experiments.
