# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:42:56 2020

@author: Raizen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes.csv')

df.info()

# fazer Oversampling ?
num_true = len(df.loc[df['Outcome'] == 1])
num_false = len(df.loc[df['Outcome'] == 0])
print("Número de Casos Verdadeiros: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Número de Casos Falsos     : {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))

df.isnull().values.any()

print("Valores Zero Glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))
print("Valores Zero BloodPressure: {0}".format(len(df.loc[df['BloodPressure'] == 0])))
print("Valores Zero SkinThickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
print("Valores Zero BMI: {0}".format(len(df.loc[df['BMI'] == 0])))

def print_Corr_Matrix(corrMatrix, size=10):
    fig, ax = plt.subplots(figsize = (size, size))
    im = ax.imshow(corrMatrix)
    im.set_clim(-1, 1)
    
    ax.set_xticks(np.arange(len(corrMatrix.columns)))
    ax.set_yticks(np.arange(len(corrMatrix.columns)))
    ax.set_xticklabels(corrMatrix.columns)
    ax.set_yticklabels(corrMatrix.columns)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(corrMatrix.shape[0]):
        for j in range(corrMatrix.shape[1]):
            text = ax.text(j, i, format(corrMatrix.iloc[i,j], '.2f'), ha='center', va='center', color='b')
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    plt.show()
    
corrMatrix = df.corr()
print_Corr_Matrix(corrMatrix, 12)

df[['Glucose','BloodPressure','SkinThickness','BMI']] = df[['Glucose','BloodPressure','SkinThickness','BMI']].replace(0, np.nan)
print(df.isnull().sum())

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.30, random_state = 42)

S_imp = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
X_treino = S_imp.fit_transform(X_treino)
X_teste = S_imp.fit_transform(X_teste)

param_grid = {'criterion': ['entropy', 'gini'],
              'max_depth': range(2,30,2),
              'min_samples_leaf': range(2,10,2),
              'min_impurity_decrease': np.linspace(0,0.5,10)}
dtc = DecisionTreeClassifier()
gs = GridSearchCV(dtc, param_grid=param_grid)
gs.fit(X_treino, y_treino)

gs.best_score_

gs.best_estimator_