import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import eig
# %matplotlib inline
# Here we are using inbuilt dataset of scikit learn
from sklearn.datasets import load_breast_cancer
  
# instantiating
cancer = load_breast_cancer()
  
# creating dataframe
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
  
# checking head of dataframe
print(df.head())
df.to_csv('cancerdata.csv')

# STEP 1 Standardization of the data
from sklearn.preprocessing import StandardScaler
  
scalar = StandardScaler()
  
# fitting
scalar.fit(df)
scaled_data = scalar.transform(df)
scaleddata_df=pd.DataFrame(scaled_data)
scaleddata_df.to_csv('standardized_data_cancerdata.csv')

# STEP 2 covariance Matrix of standardized data
print(scaled_data[0])
scaleddata_covmat=np.cov(np.transpose(scaled_data))
scaleddata_covmat_df=pd.DataFrame(scaleddata_covmat)
scaleddata_covmat_df.to_csv('cov_mat_scaleddata.csv')

#STEP 3 eigen values and eigen vectos of covariance matric
eigen_val, eigen_vect= eig(scaleddata_covmat)
print(eigen_val)
eigen_vect_df=pd.DataFrame(eigen_vect)
eigen_vect_df.to_csv('cancerdata_covmat_eigen_vec.csv')
