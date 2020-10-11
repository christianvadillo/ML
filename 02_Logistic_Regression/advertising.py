# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:59:21 2020

@author: 1052668570

fake advertising data set, indicating whether or not a particular internet user
clicked on an Advertisement. We will try to create a model that will predict
whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

'Daily Time Spent on Site': consumer time on site in minutes
'Age': cutomer age in years
'Area Income': Avg. Income of geographical area of consumer
'Daily Internet Usage': Avg. minutes a day consumer is on the internet
'Ad Topic Line': Headline of the advertisement
'City': City of consumer
'Male': Whether or not consumer was male
'Country': Country of consumer
'Timestamp': Time at which consumer clicked on Ad or closed window
'Clicked on Ad': 0 or 1 indicated clicking on Ad

"""
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import cufflinks as cf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")

data = pd.read_csv(r'../data/Regression/advertising.csv')

# Exploring data
data.head(1).T
data.info()
statistics = data.describe()

# Visual exploring
sns.heatmap(data.corr(), cmap='viridis', annot=True)

plt.hist(data['Age'], bins=30)
sns.distplot(data['Daily Time Spent on Site'], bins=30)
sns.jointplot(x='Age', y='Area Income', data=data, kind='hex')
sns.jointplot(x='Age', y='Daily Internet Usage', data=data, kind='kde', color='red')
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=data, kind='hex', color='red')
sns.pairplot(data, hue='Clicked on Ad', diag_kind='hist', palette='Pastel1', height=1.8, aspect=1, plot_kws={'s': 14, 'alpha': 0.3})
# Missing values?
sns.heatmap(data.isnull(), cmap='viridis', yticklabels=False)


# =============================================================================
# Spliting data
# =============================================================================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X.drop(['Timestamp', 'Country', 'City', 'Ad Topic Line'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================================================
# Logistic Regression
# =============================================================================
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# =============================================================================
# Evaluating model
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
metrics_report = classification_report(y_test, y_pred)
print(metrics_report)
