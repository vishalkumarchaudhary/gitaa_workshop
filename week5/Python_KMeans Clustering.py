"""
K-means clustering
"""

### Libraries to be invoked 

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# Library for invoking cluster
from sklearn import cluster

# reading the file 
data = pd.read_csv('tripDetails.csv')

data.drop(['TripID'],axis = 1,inplace = True)

######### Summarizing - descriptive statistics

######### Rows - samples and Columns - Features

data.head() # print the first 5 rows

data.info() # concise summary (includes memory usage as of pandas 0.15.0)

data.isnull().sum() # missing values in each column 

data.describe() # description of data

data.describe(include = 'all')

###########################
# Dropping columns and scaling
###########################

# Storing the features (names)
features = list(data.columns)
print(features)

data2 = data.copy()
data3 = StandardScaler().fit_transform(data2.values)
data3 = pd.DataFrame(data3,columns = features)

#####################################
# Choosing optimal number of clusters

distortions = []  # Empty list to store wss

for i in range(1, 11):
    km = cluster.KMeans(n_clusters=i,
                init='k-means++',
                n_init = 10,
                max_iter = 300,
                random_state = 100)
    km.fit(data3)
    distortions.append(km.inertia_)

#Plotting the K-means Elbow plot
plt.figure(figsize = (7,7)) 
plt.plot(range(1,11), distortions, marker='o')
plt.title('ELBOW PLOT')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


########################
# K-means Clustering
########################

# Trying different number of clusters
km2 = cluster.KMeans(n_clusters=2, random_state= 100).fit(data3)
labels2 = km2.labels_
print(labels2)

data3['labels'] = labels2
data3['labels'] = data3['labels'].astype('str')

sns.pairplot(data3,x_vars = features,y_vars = features,
             hue='labels',diag_kind='kde')
plt.show()

# choosing 3 clusters
km3 = cluster.KMeans(n_clusters=3,random_state= 100).fit(data3) 
labels3 = km3.labels_
print(labels3)

data3['labels'] = labels3
data3['labels'] = data3['labels'].astype('str')

sns.pairplot(data3,x_vars = features,y_vars = features,
             hue='labels',diag_kind='kde')
plt.show()

# choosing 4 clusters
km4 = cluster.KMeans(n_clusters=4,random_state= 100).fit(data3) 
labels4 = km4.labels_
print(labels4)

data3['labels'] = labels4
data3['labels'] = data3['labels'].astype('str')

sns.pairplot(data3,x_vars = features,y_vars = features,
             hue='labels',diag_kind='kde')
plt.show()

# From the above analysis, we found that 3 clusters represent the data well
################### END OF SCRIPT ##############################################