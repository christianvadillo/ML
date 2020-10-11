# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:40:26 2020

@author: 1052668570
"""

import pandas as pd
# import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sns.set_style("whitegrid")

data = pd.read_csv(r"../Tenserflow/DATA/cancer_classification.csv")
# missing data?
data.info()
# statistic info
data.describe().T

# Visual exploration
sns.countplot(data['benign_0__mal_1'])  # relative well balance
sns.heatmap(data.corr(), cmap='jet')
# Correlation
data.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# Splitting data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# scaling
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# # Building the deep learning model [OVERFITTING CASE]
# X_train.shape
# =============================================================================

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
# For binary classification problem we use sigmoid act
model.add(Dense(1, activation='sigmoid'))  # Output

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test))

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# =============================================================================
# # Building the deep learning model [EARLY STOP]
# # Using EarlyStopping for tracking validation loss
# =============================================================================
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
# For binary classification problem we use sigmoid act
model.add(Dense(1, activation='sigmoid'))  # Output
model.compile(loss='binary_crossentropy', optimizer='adam')

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(X_train, y_train, epochs=600,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping])

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# =============================================================================
# # Building the deep learning model [EARLY STOP & DROPOUT]
# =============================================================================
# Using EarlyStopping for tracking validation loss
# Using Dropout for preventing overfitting
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))  # 50% of neurons (random) going to turn off
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))  # 50% of neurons (random) going to turn off

# For binary classification problem we use sigmoid act
model.add(Dense(1, activation='sigmoid'))  # Output
model.compile(loss='binary_crossentropy', optimizer='adam')

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(X_train, y_train, epochs=600,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping])

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# =============================================================================
# Validating model
# =============================================================================
y_pred = model.predict_classes(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
