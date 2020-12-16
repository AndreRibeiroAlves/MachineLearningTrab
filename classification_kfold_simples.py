# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:42:56 2020

@author: Raizen
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

df = pd.read_csv('diabetes.csv')
df.isnull().values.any() # Sem valores nulos para tratar
# df = df.drop("Pregnancies", axis = 1)

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# Pré Processing
from sklearn import preprocessing
X_n = preprocessing.MinMaxScaler().fit_transform(X)

# Classification

KFold = 10

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
scores = cross_val_score(model, X, y, KFold, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))

# SVM
from sklearn import svm
#model = svm.SVC(gamma=0.001, C=10000, kernel='rbf',random_state=1)
model = svm.SVC(C=1000, kernel='rbf',random_state=1)
scores = cross_val_score(model, X_n, y, KFold, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))

# RF
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(max_depth = 5, n_estimators=500, random_state=1)
model = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=1)
scores = cross_val_score(model, X, y, KFold, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))
    
# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)
scores = cross_val_score(model, X_n, y, KFold, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))
    
# MLP
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(16,32), activation='relu',
                      solver='adam', max_iter=5000, random_state=1)
scores = cross_val_score(model, X_n, y, KFold, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))



    
