# Multivariate time-series clustering of cardiopulmonary exercise test data for cardiovascular risk stratification. 
Implementation of k-medoids clustering model using dynamic time warping (DTW) on raw time series CPET tracings.

# K-medoids model configuration
number of clusters: 5  
metric = "precomputed"  
method = "pam"  
init = "k-medoids++"  
random_state = 0  

# Model Training (iCOMPEER cohort)
cpet_clustering_training.ipynb

# External Validation (FLEMENGHO cohort)
cpet_clustering_validation.ipynb

# Python
All scripts have been tested with python version 3.9  

# Libraries
To install the required packages run "pip install -r requirements.txt". 

Some of the major libraries are:   
scikit-learn==1.4.0  
scipy==1.12.0  
scikit-learn-extra==0.3.0  
numpy==1.26.4  
