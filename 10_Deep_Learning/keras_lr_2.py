# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:32:39 2020

@author: 1052668570

https://www.kaggle.com/harlfoxem/housesalesprediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
sns.set_style("whitegrid")

data = pd.read_csv(r"../Tenserflow/DATA/kc_house_data.csv")

# missing data?
data.info()
data.isnull().sum()
sns.heatmap(data.isnull().T, cbar=False, xticklabels=False, cmap="viridis")

# statistic info
stats = data.describe().T

# Visual exploration
sns.distplot(data['price'])  # y label exploration
sns.countplot(data['bedrooms'])  # categorical 
sns.boxplot(x='bedrooms', y='price', data=data, orient="v", palette="Set2")  # outlier
sns.countplot(data['floors'])  # categorical 
sns.countplot(data['grade'])  # categorical 
sns.countplot(data['yr_built'])  # categorical 

# Correlation
plt.figure(figsize=(10,8))
sns.heatmap(data.iloc[:,1:].corr(), annot=True, cmap='viridis')
plt.tight_layout()

sns.scatterplot(data['price'], data['sqft_living'], s=50, alpha=0.3)


# Looking if the location (long and latitud) have relation with the price
sns.scatterplot(x='long', y='lat', data=data, hue='price')

# Excluding de 10% most expensive houses
data.sort_values(by='price', ascending=False).head(10)
non_top_1_perc = data.sort_values(by='price', ascending=False).iloc[216:]

sns.scatterplot(x='long', y='lat', data=non_top_1_perc, 
                hue='price', palette='magma', alpha=0.3,
                edgecolor=None, size=np.log(data['price'])/2)

sns.boxplot(y='price', x='waterfront', data=data)

# =============================================================================
# Feature engineering
# =============================================================================
data.drop('id', inplace=True, axis=1)
data['date'] = pd.to_datetime(data['date'])  # Changing to date type

# Extracting year and month for analysis
data['year'] = data['date'].apply(lambda date: date.year)
data['month'] = data['date'].apply(lambda date: date.month)

# there are a trend based in the month of sell?
sns.boxplot(x='month', y='price', data=data)
data.groupby(by='month').mean()['price'].plot()

# there are a trend based in the year of sell?
sns.boxplot(x='year', y='price', data=data)
data.groupby(by='year').mean()['price'].plot()

# Dropping off date
data.drop('date', inplace=True, axis=1)

# Is zipcode useful as categorical variable?
data['zipcode'].value_counts()

# Dropping of zipcode because 70 categories is too much
data.drop('zipcode', inplace=True, axis=1)

# Is yr_ronovated useful as categorical variable?
data['yr_renovated'].value_counts()
# Categorizig 

# =============================================================================
# Splitting data
# =============================================================================
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# =============================================================================
# scaling
# =============================================================================
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
# Building the deep learning model
# =============================================================================
X_train.shape  # 19 features

model = Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))  # price output 

model.compile(optimizer='adam', loss='mse')  # For regression model
# Training the model using batch sizes
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=128, epochs=400)


# Loss decreased?
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# Evaluating data
model.evaluate(X_test, y_test, verbose=0)  # model loss on test data
model.evaluate(X_train, y_train, verbose=0)  # model loss on training data

y_pred = model.predict(X_test).flatten()

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"Explained variance: {explained_variance_score(y_test, y_pred)}")

plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, 'r', label='True')
plt.legend()