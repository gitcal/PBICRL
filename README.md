# PBICRL
_Preference-Based Bayesian Inverse Constraint Reinforcement Learning_ (PBICRL) is a Bayesian approach that infers constraints from demonstrations. The likelihood function is based on a modification of the Bradley-Terry model that allows it to compensate for different margins among the preferences.



# Requirements
The code was written in Python 3.8.13 \
To install the requirements: \
pip install -r requirements.txt

# Contents

Each folder contains the code for the four simulation environments used in the paper. You can run the code by simply running run_experiments.sh. The data files containting the demonstrations 
can be downloaded using the following link. The data files should be saved in the corresponding data folder for each environment.
https://drive.google.com/drive/folders/1YKynJct0_ZeBkZCNKA7L2OGFtMs6v1VW?usp=sharing







If you find this code and paper relevant to your work, you can cite it as follows:

@article{papadimitriou2024bayesian,\
  title={Bayesian Constraint Inference from User Demonstrations Based on Margin-Respecting Preference Models},\
  author={Papadimitriou, Dimitris and Brown, Daniel S},\
  journal={arXiv preprint arXiv:2403.02431},\
  year={2024}\
}


