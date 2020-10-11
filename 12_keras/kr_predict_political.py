# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:22:18 2020

@author: 1052668570


Predict political party based on votes
As a fun little example, we'll use a public data set of how US congressmen 
voted on 17 different issues in the year 1984. Let's see if we can figure out 
their political party based on their votes alone, using a deep neural network!

For those outside the United States, our two main political parties are 
"Democrat" and "Republican." In modern times they represent progressive and
 conservative ideologies, respectively.

Politics in 1984 weren't quite as polarized as they are today, but you should
 still be able to get over 90% accuracy without much trouble.

Since the point of this exercise is implementing neural networks in Keras,
 I'll help you to load and prepare the data.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



feature_names = ['party','handicapped-infants', 'water-project-cost-sharing', 
                 'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                 'el-salvador-aid', 'religious-groups-in-schools',
                 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                 'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                 'education-spending', 'superfund-right-to-sue', 'crime',
                 'duty-free-exports', 'export-administration-act-south-africa']

data = pd.read_csv('..\data\classification\house-votes-84.data.txt', na_values=['?'], names=feature_names)

data.head(1).T
sns.countplot(data['party'], hue=data['crime'])


# =============================================================================
# Missing values? 
# =============================================================================
sns.heatmap(data.T.isnull(), xticklabels=False, cmap='viridis')
data.isnull().sum() / len(data) * 100

sns.countplot(data['party'], hue=data['export-administration-act-south-africa'])

#dropping out missing values
data.dropna(inplace=True)
sns.heatmap(data.T.isnull(), xticklabels=False, cmap='viridis')

# =============================================================================
# Translating categoricals to numbers
# =============================================================================
data.replace(['y', 'n'], [1, 0], inplace=True)
data.replace(['democrat', 'republican'], [1, 0], inplace=True)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# =============================================================================
# Splitting data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.2)

# =============================================================================
# Setting up the model
# =============================================================================
X_train.shape
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)


model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
          metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=1000,
          batch_size=64,
          verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping])

loss_df = pd.DataFrame(model.history.history)
loss_df[['loss', 'val_loss']].plot()

# =============================================================================
# Evaluating
# =============================================================================
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# =============================================================================
# Using model as function and CV
# =============================================================================
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


estimator = KerasClassifier(build_fn=create_model, epochs=300,
                            batch_size=64, verbose=2)

cv_scores = cross_val_score(estimator, X_train, y_train, cv=10)

cv_scores.mean()
