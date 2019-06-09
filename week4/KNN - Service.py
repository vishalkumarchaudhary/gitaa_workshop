# -*- coding: utf-8 -*-
"""
KNN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  

# import library for plotting
import matplotlib.pyplot as plt


# Reading the data set
service_data=pd.read_csv('Service.csv')

"""
unique class
"""
print(np.unique(service_data['Service']))

#####################
# Data preprocessing
#####################

# Renaming the service to 0,1
service_data['Service']=service_data['Service'].map({'No':0, 
                         'Yes':1})

service_data_columns_list=list(service_data.columns)

features=list(set(service_data_columns_list)-set(['Service']))

y=service_data['Service'].values
#

x = service_data[features].values

train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.3, random_state=0)

#data scaling
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_x)
# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

"""
KNN classification
"""

# Storing the K nearest classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 2)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
confusion_matrix = confusion_matrix(test_y, prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)


print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of K value on classifier
"""
Misclassified_sample = []
# Calculating error for K values between 1 and 40
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)

# Plotting the effect of K 
plt.figure(figsize = (7,7))
plt.plot(range(1,20,1),Misclassified_sample, 
         color='red',linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)

plt.title('Effect of K value on Misclasification')  
plt.xlabel('K Value')  
plt.ylabel('Misclassified samples')  
plt.show()

