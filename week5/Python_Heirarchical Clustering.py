# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
### **Hierarchical (Agglomerative) Clustering**

### Loading required modules
# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
### Let us first explore the given data before using the clustering algorithms 

# In[2]:
data = pd.read_csv('tripDetails.csv')

data.drop(['TripID'],axis = 1,inplace = True)
data.head()
data.info()


# In[3]:
features = list(data.columns)
print(features)

# Reminding ourselves of the units of the data-

# 'Brakes'        : 'counts',
# 'Honking'       : 'counts',
# 'IdlingTime'    : 'mins',
# 'MaxSpeed'      : 'kmph',
# 'MostFreqSpeed' : 'kmph',
# 'TripDuration'  : 'mins',
# 'TripLength'    : 'kms'

# In[6]:

# #### A look at relationship between input features - correlation
correlation = data.corr()
correlation

# From correlation matrix, we see that
# *TripLength*,*MaxSpeed*,*MostFreqSpeed* are highly correlated.
# These might be competitive variables in clustering.

# In[8]:

# #### Visualizing scatter of the data
# Since we have more than 3 features, 
# we can't visualize the scatter of the given data points 
# with all features considered at once.
# We will circumvent this by making scatter plots for each pair of features.

sns.pairplot(data)
plt.show()

# ### Observation about scatter
#  - We see that few clusters are spherically distributed and few are elliptically distributed.
#  - Also there exist different number of clusters (2, 3, 4, 5) for different pair combination of features.
#  - Few clusters are compact while others are not.
#  - In most of the scatter plots (subplots) above, we see that there are 3 candidate clusters 
#  (based on compactness and isolation).

# In[9]:

# ## **Scaling :** Important step in every Machine Learning problem -
#  - To avoid giving undue advantage to some features
#  - which are expressed in some particular units, 
#  - whose magnitude might be higer than some other feature
#  - variable (due to choice of units), scaling all features, 
#  - so that they are numerically of same order of magnitude, is **essential**.

# ### Hierarchical clustering using sklearn from python

# In[10]:

data2 = data.copy()
data3 = StandardScaler().fit_transform(data2.values)
data3 = pd.DataFrame(data3,columns = features)

# In[11]:

h_cluster = AgglomerativeClustering(n_clusters=3, 
                                    affinity='euclidean',
                                    linkage='ward').fit(data3)

labels = h_cluster.labels_


# In[12]:
### Converting labels to strings and plotting pair-wise plot
data3['labels'] = labels
data3['labels'] = data3['labels'].astype('str')

sns.pairplot(data3,x_vars = features,y_vars = features,
             hue='labels',diag_kind='kde')
plt.show()


# Again we see that for all pairs of features, the points have been well clustered.
# But to visualize the hierarchy structure (dendogram), we will use the hierarchical clustering 

# In[13]:

# #### Raw dendrogram
# With pure leaves (each leaf of dendrogram (tree like structure), represents a sample datapoint)

linkage_matrix = linkage(data3, 'ward')
figure = plt.figure(figsize=(14, 10))
dendrogram(linkage_matrix,color_threshold=3)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
# Notes: 
# - Each leave is a sample point.
# - They have been associated to each other according to closeness,
# - into clusters, until all points are in one cluster.
# 
# - But that is not easy to visualize.
# - Let us fix the number of clusters we want to see at bottom most level to be atmost 3.

# In[14]:

figure = plt.figure(figsize=(14,10))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,                   # fixing 'p'
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True),  # to get a distribution impression in truncated branches
        
plt.title('Hierarchical Clustering Dendrogram (Ward, aggrogated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
plt.show()


#  - The numbers in parenthesis indicate the number of sample points in that particular cluster.
#  - Numbers without parenthesis means only one sample point exists in that particular cluster
#  - and the number is the serial number (row index) tag asssociated with a trip.
#  - The leaves with single or fewer samples in them also give an idea of these points somehow being distinct from others.
#  - The scale on y axis represents the dissimilarity measure (distance) between different clusters possible at that level
#  - Horizontal lines are connections (crictical measures) made to agglomerate the smaller clusters together.  

# #### Finally, let us check the pair plots after choosing to have 3 clusters

# In[15]:
k=3
labels = fcluster(linkage_matrix, k,
                  criterion='maxclust')
data3['labels'] = labels
data3['labels'] = data3['labels'].astype('str')

sns.pairplot(data3,x_vars = features,y_vars = features,
             hue='labels',diag_kind='kde')
plt.show()


# #### What might these clusters represent?
# The centroids of clusters are representative of points in the clusters.
# So, let us look at the centroids -

# In[16]:

c_df = pd.concat([data[data3['labels']=='1'].mean(), 
                  data[data3['labels']=='2'].mean(), 
                  data[data3['labels']=='3'].mean()],
                  axis=1)
c_df.columns = ['cluster1','cluster2','cluster3']
c_df

# ### Observations:
#  - Cluster1 is distinguised by comparatively very high values for 
#    *TripLength*, *MaxSpeed*, *MostFreqSpeed*, *TripDuration* and low *IdlingTime*
#  - This is indicative of *intercity travel*.

#  - *MaxSpeed* is higher for cluster2 than cluster3, 
#    *TripLength* is similar, *MostFreqSpeed* of cluster2 is almost 3 times than that of cluster3.
#  - This is indicative of two regions with differing traffic densities
#    and it seems like it is higher for cluster3 than for cluser2.

#  - Looking at *TripDuration*, we are accumulating evidence to support our intuition. 
#  - Indeed the *TripDuration * is 2.75 times higher. Furthermore, huge number of times, 
#  - brakes have been applied for trips in cluster3.
 
#  - *IdlingTime* and number of *Honking* in each trip, also conform with our 
#  - experience with traffic in city center and in the outskirts and peripheral of the city.
#  - Hence cluster2 is indicative of *suburban* and cluster3 is indicative of 'city' trips.
#  
# Let us assign these names to the clusters -

# In[17]:

triptype        = ['intercity','suburban','city']
data3['labels'] = labels
data3['labels'] = data3['labels'].map({1:triptype[0],2:triptype[1],3:triptype[2]})

###############################################################################################