# -*- coding: utf-8 -*-5
"""

@author: GITAA
"""
# =============================================================================
# LINEAR REGRESSION
# =============================================================================
###################################### Required modules #######################

# Pandas - to work with data frames
import pandas as pd

# Numpy - to work with numerical computations
import numpy as np

# Sklearn - package to split data into train & test
from sklearn.model_selection import train_test_split

# Sklearn - package to use linear regression
from sklearn.linear_model import LinearRegression

# Sklearn - package to standardize data => z=(xi-mu)/sigma
from sklearn.preprocessing import StandardScaler

# Sklearn - package for performance metric 
from sklearn.metrics import mean_squared_error

# seaborn - package for visualization
import seaborn as sns

# =============================================================================
# IMPORTING DATA
# =============================================================================
data = pd.read_csv('ToyotaCorollaNew.csv')

data2=data.copy()
#"""
#Exploratory data analysis:
#
#1.Data information (types and missing value) and statistics
#2.Data preprocessing (Missing value,handling categorical variables etc)
#3.Data visualization
#"""
# ==================== To get data information ================================
print(data.info()) 

# ==================== To get 5 number summary and counts================================
print(data.describe()) 

data['FuelType'].value_counts() # Frequency table for 'Fuel Type'

# ==================== To get data types ================================
data.dtypes
data['Automatic'].dtype
data['MetColor'].dtype
data['FuelType'].dtype

# ==================== Cross Tables ================================

pd.crosstab(data['FuelType'],data['Automatic'],
            dropna=True)


# ==================== Checking for missing values ============================
data.isnull().sum()
print('Data columns with null values:\n', 
      data.isnull().sum()) 


# ==================== Replacing 'Age' with mean ==============================
data['Age'].mean()
data['Age'].fillna(data['Age'].mean(), 
    inplace = True)
data['Age'].isnull().sum()

# ==================== Replacing 'Fuel Type' with mode ========================
data['FuelType'].value_counts() # Frequency table for 'Fuel Type'

data['FuelType'].mode()  
# To get the mode value of 'Fuel Type'

data['FuelType'].mode()[0]     #or
data['FuelType'].value_counts().index[0]


data['FuelType'].fillna(data['FuelType'].mode()[0], 
    inplace = True)
data['FuelType'].isnull().sum()

# ==================== Replacing 'MetColor' with mode ========================

data['MetColor']=data['MetColor'].astype('object')


data['MetColor'].value_counts() # Frequency table for 'MetColor'

# replacing MetColor with mode
data['MetColor'].fillna(data['MetColor'].mode()[0], inplace = True)

## Check for missing data after filling values 
print('Data columns with null values:\n', data.isnull().sum())


# ==================== Imputation using lambda functionss ========================

#only categorical
#data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))
#only numerical
#data = data.apply(lambda x:x.fillna(x.mean()))

data2 = data2.apply(lambda x:x.fillna(x.mean()) \
                  if x.dtype=='float' else \
                  x.fillna(x.value_counts().index[0]))

# =============================================================================
# Dropping a column
data.drop('Unnamed: 0',axis=1,inplace=True)

# =============================================================================
# CORRELATION
# =============================================================================
# Create new variable to store data without categorical variables
data['Automatic']=data['Automatic'].astype('object')
data['MetColor']=data['MetColor'].astype('object')


data_new = data.select_dtypes(exclude=[object])
print(data_new.shape)

# Finding the correlation between numerical variables
data_stat = data_new.corr()
print(data_stat)


print(round(data_stat,2)) # Rounding of decimals

# =============================================================================
# DUMMY VARIABLE ENCODING
# =============================================================================
# Convert string into dummy variable

data=pd.get_dummies(data,drop_first=True) 
#true- it removes cng
# drop_first => whether to get k-1 dummies out of k categorical levels by 
# removing the level which is less in frequency.

# =============================================================================
# Data visualization: Pair plot (output,input)
# =============================================================================
df_select = data.select_dtypes(include = ['float64', 
                                          'int64'])
for i in range(0, len(df_select.columns), 3):
    sns.pairplot(data=df_select,
                x_vars=df_select.columns[i:i+3],
                y_vars=['Price'])

# =============== Regression modeling starts from here ========================

# Storing the column names in variables 
features = list(set(data.columns)-set(['Price']))
target   = list(['Price'])

print(features)
print(target)

# =============================================================================

# Separating out the features
x = data.loc[:, features].astype(float)   # remove .values and put .astype(float)

y = data.loc[:,target].astype(float)      #.astype(float)

# We are splitting test & train as 30% and 70%
train_x, test_x, train_y, test_y = train_test_split\
                                   (x,y,test_size=0.3,\
                                    random_state=1)
################### base line model ####################################

"""
We are making a base model by using test data mean value
to keep as benchmark to compare with our regression model
"""

# finding the mean for test data value
base_pred = np.mean(test_y)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(test_y))

# finding the RMSE
base_root_mean_square_error = (mean_squared_error\
                               (test_y, base_pred))**0.5
                               
print(base_root_mean_square_error)

#
##############################################################

# Data Scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()
##Standardize features by removing the mean and scaling to unit variance
"""
 Centering and scaling happen independently on each feature 
by computing the relevant statistics on the samples in the 
training set. Mean and standard deviation are then stored to 
be used on later data using the transform method.
Fit on training set only.
"""
scaler_x.fit(train_x) # .fit computes the mean and std to be used for later scaling
scaler_y.fit(train_y)
# Apply transform to both the training set and the test set.

# performs standardization by centering and scaling
train_x = scaler_x.transform(train_x)
test_x  = scaler_x.transform(test_x)
train_y = scaler_y.transform(train_y)

# =============================================================================
# LINEAR REGRESSION
# =============================================================================
# Building a regression model by making intercept 0, 
#since data is scaled
lgr=LinearRegression(fit_intercept=False)

# Fitting the train data for linear model
x=lgr.fit(train_x,train_y)
x.coef_

# Predicting from the test data
predicted_y_lgr = lgr.predict(test_x)

# Rescaling the data to original 
y_predicted_in_original_scale = \
            scaler_y.inverse_transform(predicted_y_lgr)

# RMSE of the linear model
lr_rmse = (mean_squared_error(test_y,\
                    y_predicted_in_original_scale))\
                              **0.5

print(lr_rmse)

from matplotlib import pyplot
pyplot.plot(test_y,'b*')
pyplot.plot(y_predicted_in_original_scale,'r*')

"""
A common method of measuring the accuracy of regression models is to use the R2 statistic.

The R2 statistic is defined as follows:

#    R2=1–RSS/TSS
#
#    The RSS (Residual sum of squares) measures the variability 
#    left unexplained after performing the regression

#    The TSS measues the total variance in Y

#    Therefore the R2 statistic measures proportion of variability 
#    in Y that is explained by X using our model 
"""

def r_square(test_y,predicted_y):
    RSS=np.sum((predicted_y-test_y)**2)      #SSE
    TSS= np.sum((test_y-np.mean(test_y))**2) #SST
    r_2=1-(RSS/TSS)
    return r_2
       
# Passing arguments to function    
r2_test_data = r_square(test_y,y_predicted_in_original_scale)
print(r2_test_data)
## Note the difference in argument order

# =============================================================================
# LINEAR REGRESSION USING 'statsmodels' PACKAGE
# =============================================================================
import statsmodels.api as sm
model = sm.OLS(train_y, train_x).fit() ## sm.OLS(output, input)
predictions = model.predict(test_x)
y_predicted_in_original_scale2 = scaler_y.inverse_transform(predictions)

## Print out the statistics
model.summary()

## RMSE of the linear model
lr_rmse2 = (mean_squared_error(test_y,y_predicted_in_original_scale2))**0.5
print(lr_rmse2)

# =============================================================================
# END OF SCRIPT
# =============================================================================
