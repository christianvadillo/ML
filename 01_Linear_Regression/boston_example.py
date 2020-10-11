# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:53:51 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

boston = load_boston()
boston.keys()

# Description
print(boston['DESCR'])

# Data
print(boston['data'])

# Target
print(boston['target'])

X = pd.DataFrame(columns=boston['feature_names'], data=boston['data'])
y = boston['target']

# Exploring plots
# sns.pairplot(X)
# X.describe()
# sns.heatmap(X.corr(), cmap='BrBG', annot=True)
# plt.tight_layout()

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting
y_pred = lm.predict(X_test)

betha = lm.intercept_
coef = pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coeff'])

# Residuals
residuals = abs(y_test - y_pred)
plt.plot(residuals)
sns.distplot(residuals)

# Metrics
r = lm.score(X_test, y_test)

mae = mean_absolute_error(y_test, y_pred)  # average error
mse = mean_squared_error(y_test, y_pred)  # punish larger errors
rmse = np.sqrt(mse)  # interpretable mse with no y-units



plt.scatter(range(len(X_test)), y_pred, alpha=0.3, s=y_pred*3)
plt.scatter(range(len(X_test)), y_test, alpha=0.8, s=y_test*3)
plt.scatter(y_test, y_pred)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title("True vs Predicted")