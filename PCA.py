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

#Step 4 sort the eigen values in descending order 
eigen_vect_df.to_csv('cancerdata_covmat_eigen_vec.csv')


#Step 5 sort the eigen values in descending order

#step6  calcualte  val1 = eigenvalue/SUM(all eigen values)

# step7 calculate the cummulative summation of val1

# step8 select the top eigen values whose summation is  .97 to .99
# and discard all the below eigen value and correcponding eigen vector
# as 97 to 99 percent information about the data is stored in selected eigen values

# most of te cases top 30 to 40 % eigen values contain more than 95% information
# rrest of the eigen values and corresponding vectors and features can be removed from data



