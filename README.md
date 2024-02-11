# Multivariate time-series clustering of cardiopulmonary exercise test for cardiovascular risk stratification. 

# K-medoids model configuration
number of clusters: 5  
metric = "precomputed"  
method = "pam"  
init = "k-medoids++"  
random_state = 0  

# Training and Validation on FLEMENGHO Cohort
cpet_clustering_training.ipynb

# External Validation on EPOGH Cohort
cpet_clustering_validation.ipynb

# Python
all scripts have been tested with python version 3.9 environment

# Libraries
To install the required packages run "pip install -r requirements.txt". 

Some of the major libraries are:   
scikit-learn==1.4.0  
scipy==1.12.0  
scikit-learn-extra==0.3.0  
numpy==1.26.4  
