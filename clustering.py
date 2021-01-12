# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:39:40 2021

@author: Raizen
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial as sps
from matplotlib import pyplot as plt 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.decomposition import SparsePCA, TruncatedSVD
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.cluster import MatrixLabelSpaceClusterer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

# Read Data
df = pd.read_csv("winequality-red.csv")
classes = df.iloc[:,-1].values
df_num_classes = len(set(df.iloc[:,-1].values))
df = df.drop(columns=['quality'])

# Fit and Preprocessing
#df = SparsePCA(n_components=2, random_state=1).fit_transform(df)
df = sp.sparse.coo_matrix(df)
df = preprocessing.normalize(df)
df = shuffle(df, random_state = 1)
df = sp.sparse.coo_matrix(TruncatedSVD(n_components=2, random_state=1).fit_transform(df))

# K-Means
X = df.tocsr()
n_clusters = df_num_classes

model = KMeans(n_clusters=n_clusters, random_state=1, n_jobs=-1).fit(X)
pred = model.predict(X)

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

# Classification

'''
from sklearn.preprocessing import LabelBinarizer
original_classes = LabelBinarizer().fit_transform(classes)
pred_classes = LabelBinarizer().fit_transform(pred)

clf = RandomForestClassifier()
clf.fit(df, pred_classes)
res_class = clf.predict(df)

accuracy_score = metrics.accuracy_score(res_class, original_classes)
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
