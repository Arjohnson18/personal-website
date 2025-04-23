# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:18:03 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

###---------------- HIERARCHICAL clustering -----------------###
###--------------Import and Format Data
file_path = r"C:\Users\Arjoh\Downloads\Utilities.csv"
utilities_df = pd.read_csv(file_path)
utilities_df.set_index('Company', inplace=True)
#Conversion of integer data to float to avoid a scale function warning 
utilities_df = utilities_df.apply(lambda x: x.astype('float64'))

###--------------Compute Euclidean distance
d = pairwise.pairwise_distances(utilities_df, metric='euclidean')
a = pd.DataFrame(d, columns=utilities_df.index, index=utilities_df.index)

#Normalize Data Before Distance Calculation
utilities_df_norm_1 = utilities_df.apply(preprocessing.scale, axis=0) #scikit-learn uses population standard deviation
utilities_df_norm_2 = (utilities_df - utilities_df.mean())/utilities_df.std() #pandas uses sample standard deviation

#Compute Euclidean Distance on Normalized Subset
d_norm = pairwise.pairwise_distances(utilities_df_norm_2[['Sales', 'Fuel_Cost']], metric='euclidean')
b = pd.DataFrame(d_norm, columns=utilities_df.index, index=utilities_df.index)

###--------------Hierarchical Clustering + Dendrogram Visualization
# in linkage() set argument method = ’single’, ’complete’, ’average’, ’weighted’, centroid’, ’median’, ’ward’
Z = linkage(utilities_df_norm_2, method='single')
dendrogram(Z, labels=utilities_df_norm_2.index, color_threshold=2.75)
plt.show()

Z = linkage(utilities_df_norm_2, method='average')
dendrogram(Z, labels=utilities_df_norm_2.index, color_threshold=3.6)
plt.show()

###--------------Assign Cluster Membership (Cut the Dendrogram)
#Single Linkage 
memb = fcluster(linkage(utilities_df_norm_2, method='single'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm_2.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))

#Average Linkage
memb = fcluster(linkage(utilities_df_norm_2, method='average'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm_2.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))

#Analyze Cluster Centroids (Original Data)
memb = fcluster(linkage(utilities_df_norm_2, method='single', metric='euclidean'), 6, criterion='maxclust')
centroids = {}
for key, item in utilities_df.groupby(memb):
    centroids[key] = item.mean()
    print('Cluster {}: size {}'.format(key, len(item)))
s = pd.DataFrame(centroids).transpose()
print (s)

###--------------Visualize with Heatmap
# set labels as cluster membership and utility name
utilities_df_norm_2.index = ['{}: {}'.format(cluster, state)
                           for cluster, state in zip(memb, utilities_df_norm_2.index)]
# plot heatmap
# the ’_r’ suffix reverses the color mapping to large = dark
sns.clustermap(utilities_df_norm_2, method='average', col_cluster=False, cmap='mako_r')


###---------------- NON-HIERARCHICAL clustering -----------------###
###--------------Import and Format Data
file_path = r"C:\Users\Arjoh\Downloads\Utilities.csv"
utilities_df = pd.read_csv(file_path)
utilities_df.set_index('Company', inplace=True)
utilities_df = utilities_df.apply(lambda x: x.astype('float64'))

###--------------Compute Euclidean distance
#Normalize distances
utilities_df_norm = utilities_df.apply(preprocessing.scale, axis=0)
kmeans = KMeans(n_clusters=6, random_state=0).fit(utilities_df_norm) #Apply K-Means Clustering (k=6)

#Cluster membership
memb = pd.Series(kmeans.labels_, index=utilities_df_norm.index)   #here labels are the cluster labels from 0 to 5 in our case of 6 clusters
for key, item in memb.groupby(memb):
     print(key, ': ', ', '.join(item.index))
     
###--------------Compute and Display Cluster Centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=utilities_df_norm.columns)
pd.set_option('precision', 3)  #set display option on the environment (e.g., console to display 3 decimals)
print(centroids)

###--------------Within-Cluster Sum of Squares (WCSS)
#calculate the distances of each data point to the cluster centers
distances = kmeans.transform(utilities_df_norm)

#Reduce to the minimum squared distance of each data point to the cluster centers
minSquaredDistances = distances.min(axis=1) ** 2

#Combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_}, 
    index=utilities_df_norm.index)

#Group by cluster and print information
for cluster, data in df.groupby('cluster'):
    count = len(data)
    withinClustSS = data.squaredDistance.sum()
    print(f'Cluster {cluster} ({count} members): {withinClustSS:.2f} within cluster ')

###--------------Euclidean Distance between Cluster centroids
c = pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_, metric='euclidean'))

#Parallel Coordinates Plot (Cluster Profiles)
centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
plt.figure(figsize=(7,6))
plt.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(-0.5,7.5)
plt.show()

###--------------Determine Optimal Number of Clusters (Elbow Method)
#when k is unknon, you can determine the optimal # of clusters graphically
inertia = []
for n_clusters in range(1, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(utilities_df_norm)
    inertia.append(kmeans.inertia_ / n_clusters)
    
inertias = pd.DataFrame({'n_clusters': range(1, 7), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()