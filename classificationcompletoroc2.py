# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:42:56 2020

@author: Raizen
"""

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
#https://www.discoverbits.in/371/sklearn-attributeerror-predict_proba-available-probability
#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html
#https://scikit-learn.org/dev/modules/svm.html#scores-and-probabilities
#https://github.com/scikit-learn/scikit-learn/issues/13662
#https://www.kite.com/python/answers/how-to-plot-an-roc-curve-in-python
#https://stackabuse.com/understanding-roc-curves-with-python/
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
#https://www.google.com/search?q=simple+roc+curve+python&oq=simple+roc+curve+python&aqs=chrome..69i57j33i22i29i30l7.4108j1j7&sourceid=chrome&ie=UTF-8
#https://www.google.com/search?q=roc+curve+k+fold&oq=kfold+roc&aqs=chrome.1.69i57j0i8i13i30l3.4020j0j7&sourceid=chrome&ie=UTF-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import io

from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def add_Score(key, original, predicted, scores):
    
    if key in scores:
        result = scores[key][1]
        add_Accuracy("accuracy", original, predicted, result)
        add_MCCScore("mcc", original, predicted, result)
        add_RecallScore("Recall", original, predicted, result)
        add_F1Score("F1-Score", original, predicted, result)
        scores[key][1] = result
    else:
        result = dict()
        add_Accuracy("accuracy", original, predicted, result)
        add_MCCScore("mcc", original, predicted, result)
        add_RecallScore("Recall", original, predicted, result)
        add_F1Score("F1-Score", original, predicted, result)
        scores[key] = [key, result]
    
    return scores

def add_Accuracy(key, original, predicted, metric):
    Score = metrics.accuracy_score(original.values, predicted) # Da pra adicionar outras métricas aqui também.
    if key in metric:
        metric[key][1].append(Score)
    else:
        metric[key] = [key,[Score]]
    return metric

def add_PrecisionScore(key, original, predicted, metric):
    Score = metrics.precision_score(original.values, predicted, average='binary', zero_division=0) # Da pra adicionar outras métricas aqui também.
    if key in metric:
        metric[key][1].append(Score)
    else:
        metric[key] = [key,[Score]]
    return metric

def add_MCCScore(key, original, predicted, metric):
    Score = metrics.matthews_corrcoef(original.values, predicted) # Da pra adicionar outras métricas aqui também.
    if key in metric:
        metric[key][1].append(Score)
    else:
        metric[key] = [key,[Score]]
    return metric

def add_RecallScore(key, original, predicted, metric):
    Score = metrics.recall_score(original.values, predicted, average='binary') # Da pra adicionar outras métricas aqui também.
    if key in metric:
        metric[key][1].append(Score)
    else:
        metric[key] = [key,[Score]]
    return metric

def add_F1Score(key, original, predicted, metric):
    Score = metrics.f1_score(original.values, predicted, average='binary') # Da pra adicionar outras métricas aqui também.
    if key in metric:
        metric[key][1].append(Score)
    else:
        metric[key] = [key,[Score]]
    return metric

def show_metric(Scores, Metric_Name=["all"]):
    for Score in Scores:
        metcs = list(Score[1].values())
        for metric in metcs:
            if "all" in Metric_Name or metric[0] in Metric_Name:
                metric[1] = np.array(metric[1])
                print(Score[0], metric[0]+":", round(metric[1].mean(),2), "Desvio:", round(metric[1].std(),4))

def to_pandas(Scores, Metric_Name=["all"], classifiers_len = 5, metrics_len = 4):
    list_resultados = []
    list_metrics = np.ndarray((metrics_len), dtype="object")
    list_classificadores = np.ndarray((classifiers_len), dtype="object")
    for i in range(len(Scores)):
        Score = Scores[i]
        list_classificadores[i] = Score[0]
        metcs = list(Score[1].values())
        list_res = list()
        for j in range(len(metcs)):
            metric = metcs[j]
            metric[1] = np.array(metric[1])
            list_metrics[j] = metric[0]
            list_res.append(round(metric[1].mean(),2))
        list_resultados.append(np.array(list_res))
    list_resultados = np.array(list_resultados) 
    return pd.DataFrame(list_resultados, columns=list_metrics, index=list_classificadores)
                
                

# https://techoverflow.net/2013/12/08/converting-a-pandas-dataframe-to-a-customized-latex-tabular/
def convertToLaTeX(df, alignment="c"):
    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    colFormat = ("%s|%s" % (alignment, alignment * numColumns))
    #Write header
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    columnLabels = ["\\textbf{%s}" % label for label in df.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    #Write data lines
    for i in range(numRows):
        output.write("\\textbf{%s} & %s\\\\\n"
                     % (df.index[i], " & ".join([str(val) for val in df.iloc[i]])))
    #Write footer
    output.write("\\end{tabular}")
    return output.getvalue()

# df1 = pd.read_csv('diabetes_johndasilva_kaggle.csv') #  hospital Frankfurt, Germany HFG
# df2 = pd.read_csv('diabetes.csv')
# df = pd.concat([df1,df2], axis=0)

df = pd.read_csv('diabetes_johndasilva_kaggle.csv') #  hospital Frankfurt, Germany HFG
#df = pd.read_csv('diabetes.csv')
df = shuffle(df, random_state=1)

df.isnull().values.any() # Sem valores nulos para tratar
# df = df.drop("Pregnancies", axis = 1)

# Fit Data
from sklearn.model_selection import LeaveOneOut
kf = LeaveOneOut()

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

class_pos = "Outcome"
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(class_pos, axis = 1),df[class_pos], test_size = 0.4, random_state = 1)

Scores = dict()

# Pré Processing
X_train_n = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test_n = preprocessing.MinMaxScaler().fit_transform(X_test)

# Classification

fig, ax = plt.subplots()

# Naive Bayes
# model = MultinomialNB()
model = GaussianNB()
model.fit(X_train_n, y_train)
predicted = model.predict(X_test_n)
add_Score("Naive Bayes", y_test, predicted, Scores)
metrics.plot_roc_curve(model, X_test_n, y_test, name='ROC curve Naive Bayes', alpha=0.8, lw=1.2, ax=ax)
# SVM
#model = svm.SVC(gamma=0.001, C=10000, kernel='rbf',random_state=1)
model = svm.SVC(C=2, kernel='linear',random_state=1)
model.fit(X_train,y_train)
predicted = model.predict(X_test)
add_Score("SVM", y_test, predicted, Scores)
metrics.plot_roc_curve(model, X_test, y_test, name='ROC curve SVM', alpha=0.8, lw=1.2, ax=ax)

# RF
#model = RandomForestClassifier(max_depth = 5, n_estimators=500, random_state=1)
model = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=1,
                               bootstrap = True, max_features = 'sqrt', max_depth = 5)
model.fit(X_train,y_train)
predicted = model.predict(X_test)
add_Score("RF", y_test, predicted, Scores)
metrics.plot_roc_curve(model, X_test, y_test, name='ROC curve RF', alpha=0.8, lw=1.2, ax=ax)
    
# KNN
model = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)
model.fit(X_train_n,y_train)
predicted = model.predict(X_test_n)
add_Score("KNN", y_test, predicted, Scores)
metrics.plot_roc_curve(model, X_test_n, y_test, name='ROC curve KNN', alpha=0.8, lw=1.2, ax=ax)
    
# MLP
model = MLPClassifier(hidden_layer_sizes=(16,32), activation='relu',
                      solver='adam', max_iter=5000, random_state=1)
model.fit(X_train_n,y_train)
predicted = model.predict(X_test_n)
add_Score("MLP", y_test, predicted, Scores)
metrics.plot_roc_curve(model, X_test_n, y_test, name='ROC curve MLP', alpha=0.8, lw=1.2, ax=ax)
    
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic (ROC)")
ax.legend(loc="lower right")
plt.show()

Scores = list(Scores.values())
# show_metric(Scores, ["accuracy"])
show_metric(Scores)
resultados = to_pandas(Scores)

print(convertToLaTeX(resultados))
# print(resultados.to_latex())





    
