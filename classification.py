# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:42:56 2020

@author: Raizen
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score

df = pd.read_csv('diabetes.csv')
df.isnull().values.any() # Sem valores nulos para tratar
# df = df.drop("Pregnancies", axis = 1)

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# Fit Data
class_pos = "Outcome"
X_train, X_test, X_train, y_test = train_test_split(
    df.drop(class_pos, axis = 1),df[class_pos], test_size = 0.4, random_state = 1)

# Pré Processing
from sklearn import preprocessing
X_train_n = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test_n = preprocessing.MinMaxScaler().fit_transform(X_test)

# Classification

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_n, y_train)
# model.predict(X_test_n)
# model.score(X_test_n, y_test)
scores = cross_val_score(model, X_train_n, y_train, cv = 10, scoring = "accuracy")
print("K fold Score: " + str(scores.mean()) + " Desvio: " + str(scores.std()))

# SVM
from sklearn import svm
#model = svm.SVC(gamma=0.001, C=10000, kernel='rbf',random_state=1)
model = svm.SVC(C=1000, kernel='rbf',random_state=1)
model.fit(X_train,y_train)
# model.predict(X_test_n)
model.score(X_test, y_test)

# RF
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(max_depth = 5, n_estimators=500, random_state=1)
model = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=1)
model.fit(X_train_n,y_train)
# model.predict(X_test_n)
model.score(X_test_n, y_test)
    
# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)
model.fit(X_train_n,y_train)
# model.predict(X_test_n)
model.score(X_test_n, y_test)
    
# MLP
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(16,32), activation='relu',
                      solver='adam', max_iter=5000, random_state=1)
model.fit(X_train_n,y_train)
# model.predict(X_test_n)
model.score(X_test_n, y_test)



    
