# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:38:54 2020

@author: 1052668570

Congratulations! You just got some contract work with an Ecommerce company
based in New York City that sells clothing online but they also have in-store
style and clothing advice sessions. Customers come in to the store, have
sessions/meetings with a personal stylist, then they can go home and order
either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile
app experience or their website. They've hired you on contract to help them
figure it out! Let's get started!


Data:
    We'll work with the Ecommerce Customers csv file from the company. 
    It has Customer info, such as:
        -Email
        -Address
        -Color Avatar. 
    Then it also has numerical value columns:
        Avg.Session Length: Average session of in-store style advice sessions.
        Time on App: Average time spent on App in minutes
        Time on Website: Average time spent on Website in minutes
        Length of Membership: How many years the customer has been a member.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

sns.set_style("whitegrid")
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10) # Used to visualize describe method
pd.set_option('display.width', 80)  # Used to visualize describe method

data = pd.read_csv(r'data/Regression/Ecommerce Customers.csv')

# Exploring data
data.head(2)
data.info()
data.describe().T
data.columns

# Visualizing data
sns.jointplot(data['Time on Website'], data['Yearly Amount Spent'], kind='hex')
sns.jointplot(data['Time on App'], data['Yearly Amount Spent'], kind='hex')
sns.jointplot(data['Length of Membership'], data['Yearly Amount Spent'], kind='hex')
sns.jointplot(data['Avg. Session Length'], data['Yearly Amount Spent'], kind='hex')

sns.pairplot(data, height=1.8, aspect=1, plot_kws={'s': 14, 'alpha': 0.3})

# Yearly Amount Spent vs. Length of Membership
sns.lmplot('Length of Membership', 'Yearly Amount Spent', data=data)

# Spliting data
X = data.iloc[:, 3:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting new data
y_pred = lm.predict(X_test)

# True vs pred plot
plt.scatter(y_test, y_pred)

# Evaluating the model
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Residuals
residuals = y_test - y_pred
sns.distplot(residuals, bins=50)

# Coeffecients
coeff = pd.DataFrame(index=X.columns, columns=['Coeffecient'], data=lm.coef_)
print(coeff)
