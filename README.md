# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data

# OUTPUT
<img width="1686" height="742" alt="image" src="https://github.com/user-attachments/assets/5167cf71-2a44-4f05-aac5-fab8cbd3525c" />

# CODING
data.isnull().sum()

# OUTPUT
<img width="206" height="600" alt="image" src="https://github.com/user-attachments/assets/6f129885-fc24-446a-bcb2-55b69e205513" />

# CODING
missing=data[data.isnull().any(axis=1)]
missing

# OUTPUT
<img width="1685" height="732" alt="image" src="https://github.com/user-attachments/assets/5a7b5af5-0fa3-4b65-88f3-f0610a3dc6b9" />

# CODING
data2=data.dropna(axis=0)
data2

# OUTPUT
<img width="1673" height="743" alt="image" src="https://github.com/user-attachments/assets/408f19c8-ed29-452c-abab-cf675071d36f" />

# CODING
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

# OUTPUT
<img width="413" height="256" alt="image" src="https://github.com/user-attachments/assets/282c115c-42e3-4939-930d-06e164081016" />

# CODING
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs

# OUTPUT
<img width="444" height="509" alt="image" src="https://github.com/user-attachments/assets/5c52d1e2-44a8-4ff9-a762-a518ceb22dcb" />

# CODING
data2

# OUTPUT
<img width="1696" height="571" alt="image" src="https://github.com/user-attachments/assets/9ddb0dbb-a776-46e1-b6a1-1b90acdf1731" />

# CODING
new_data=pd.get_dummies(data2, drop_first=True)
new_data

# OUTPUT
<img width="1744" height="605" alt="image" src="https://github.com/user-attachments/assets/3ba894d9-3f0b-45b4-b1e7-29aee34a8ba1" />

# CODING
columns_list=list(new_data.columns)
print(columns_list)

# OUTPUT
<img width="1197" height="397" alt="image" src="https://github.com/user-attachments/assets/9b055b65-db74-4432-8652-bd48c73a6372" />

# CODING
features=list(set(columns_list)-set(['SalStat']))
print(features)

# OUTPUT
<img width="1198" height="401" alt="image" src="https://github.com/user-attachments/assets/54062299-923e-48fa-98f8-d81d82c3d426" />










# RESULT:
       # INCLUDE YOUR RESULT HERE
