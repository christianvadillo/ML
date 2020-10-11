# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:42:50 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from sklearn.datasets import load_iris
sns.set_style("whitegrid")

iris = load_iris()
iris.keys()
print(iris['DESCR'])

data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
# =============================================================================
# Exploratory
# =============================================================================
data.head(1).T
data.info()
data.describe()

# =============================================================================
# Splitting data
# =============================================================================
# =============================================================================
X = data.values
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# =============================================================================
# SVM Model (Classification)
# =============================================================================
svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# Using GridSearch to find best hyperparameters
# =============================================================================
param_grid = {'C':  [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_

# =============================================================================
# Evaluating grid model
# =============================================================================
gs_predict = grid.predict(X_test)
print(confusion_matrix(y_test, gs_predict))
print(classification_report(y_test, gs_predict))
