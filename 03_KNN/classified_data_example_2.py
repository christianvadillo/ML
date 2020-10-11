# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:55:24 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Use StandardScaler if you know the data distribution is normal.
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sns.set_style("whitegrid")

data = pd.read_csv('data/classification/KNN_Project_Data')

# =============================================================================
# Exploratory
# =============================================================================
data.head(1).T
data.dtypes  # One string
stats = data.describe().T

sns.countplot(data['TARGET CLASS'])  # balanced
# sns.pairplot(data, hue='TARGET CLASS') # data looks normal

# =============================================================================
# Normalizing data
# Because the scale of the variables matters for knn
# =============================================================================
sc = StandardScaler()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# KNN Model
# =============================================================================
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# =============================================================================
# Evaluating model
# =============================================================================

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# Using Elbow method to select n_neighbors
# =============================================================================
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.plot(error_rate, linestyle='--',color='b', 
         marker='o', markerfacecolor='red')
plt.xlabel("K")
plt.ylabel("Error rate")
plt.title("Error Rate vs K value")

# =============================================================================
# KNN Model with K=16
# =============================================================================
model = KNeighborsClassifier(n_neighbors=16)
model.fit(X_train, y_train)

# =============================================================================
# Evaluating model
# =============================================================================

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
