# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:39:40 2021

@author: Raizen
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial as sps
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.decomposition import SparsePCA, TruncatedSVD, FactorAnalysis, DictionaryLearning, FastICA, KernelPCA

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, AffinityPropagation, AgglomerativeClustering, FeatureAgglomeration
from sklearn import cluster, covariance, manifold
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.cluster import MatrixLabelSpaceClusterer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier



# Read Data
df = pd.read_csv("winequality-red.csv")
classes = df.iloc[:,-1].values
df_num_classes = len(set(df.iloc[:,-1].values))
df = df.drop(columns=['quality'])

# Fit and Preprocessing
# df = KernelPCA(n_components=2, random_state=1).fit_transform(df)
# df = FastICA(n_components=2, random_state=1).fit_transform(df)
# df = DictionaryLearning(n_components=2, random_state=1,n_jobs=-1).fit_transform(df)
# df = FactorAnalysis(n_components=2, random_state=1).fit_transform(df)
df = SparsePCA(n_components=2, random_state=1).fit_transform(df)
df = sp.sparse.coo_matrix(df)
df = preprocessing.normalize(df)
df = shuffle(df, random_state = 1)
# df = sp.sparse.coo_matrix(TruncatedSVD(n_components=2, random_state=1).fit_transform(df))

# Parse
X = df.tocsr()
n_clusters = df_num_classes

'''
# K-Means
model = KMeans(n_clusters=n_clusters, random_state=1, n_jobs=-1).fit(X)
pred = model.predict(X)
'''

'''
# DBSCAN
model = DBSCAN(eps=0.078, min_samples=5, metric='euclidean',
               metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
pred = model.fit_predict(X)
'''

'''
# Mean Shift
bandwidth = estimate_bandwidth(X.toarray(), quantile=.1, n_samples=500, random_state=1, n_jobs=-1)
model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1).fit(X.toarray())
pred = model.labels_
'''

'''
# Affinity Propagation
# edge_model = covariance.GraphicalLassoCV().fit(X.toarray())
# node_position_model = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=2, eigen_solver='dense')
# embedding = node_position_model.fit_transform(X.toarray().T)
model = AffinityPropagation().fit(X)
pred = model.labels_
# _, pred = cluster.affinity_propagation(edge_model.covariance_)
'''

# AglomerativeClustering
model = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'euclidean', linkage = 'ward')
pred = model.fit_predict(X.toarray())

'''
# FeatureAgglomeration
model = FeatureAgglomeration(n_clusters = 2)
pred = model.fit_transform(X.toarray())
'''

'''
# Divisive
# https://github.com/ronak-07/Divisive-Hierarchical-Clustering/blob/master/Divisive.py
'''

# Evaluate
Score = metrics.silhouette_score(X, model.labels_)

# Show
X_plot = X.toarray()

plt.scatter(X_plot[:,0], X_plot[:,1], s=5)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c='yellow')
plt.show()

sps.voronoi_plot_2d(sps.Voronoi(model.cluster_centers_))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=pred, s=1, cmap='magma')
plt.show()

graph = dendrogram(linkage(X_plot, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Data')
plt.ylabel('Distância Euclidiana')

plt.scatter(X_plot[:,0],X_plot[:,1], c=pred, cmap='rainbow')
plt.show()

# Classification

'''
from sklearn.preprocessing import LabelBinarizer
original_classes = LabelBinarizer().fit_transform(classes)
pred_classes = LabelBinarizer().fit_transform(pred)

clf = RandomForestClassifier()
clf.fit(df, pred_classes)
res_class = clf.predict(df)

accuracy_score = metrics.accuracy_score(res_class, original_classes)
accuracy_score = metrics.accuracy_score(original_classes, pred_classes)
'''

'''
from sklearn.cluster import MiniBatchKMeans
total_clusters = 3
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# Fitting the model to training set
kmeans.fit(x)

a = kmeans.labels_
'''
'''
# construct base forest classifier
base_classifier = RandomForestClassifier(n_estimators=102, n_jobs=-1)

# setup problem transformation approach with sparse matrices for random forest
problem_transform_classifier = LabelPowerset(classifier=base_classifier,
    require_dense=[False, False])

# setup the clusterer
clusterer = MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=3, random_state=0, n_jobs=-1))

# setup the ensemble metaclassifier
classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

# train
classifier.fit(x, y)

# KMeans = KMeans(n_clusters=19, random_state=0, n_jobs=-1).fit(x)
# pred = KMeans.predict(x)

# predict
predictions = classifier.predict(x_valid)
predictions_ = predictions.toarray()
from sklearn.metrics import accuracy_score
accuracy_score(y, predictions_)
'''
'''
reference_labels = retrieve_info(kmeans.labels_, y)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]
'''
