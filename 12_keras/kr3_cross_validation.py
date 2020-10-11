# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:54:33 2020

@author: 1052668570
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier

data = pd.read_csv('../data/Regression/weight-height.csv')


# =============================================================================
# model function to wrapp
# =============================================================================
def build_model():
    # Logistic regression model
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.1),
                  'binary_crossentropy',
                  metrics=['accuracy'])
    return model


# =============================================================================
# Creating a wrapper model
# =============================================================================
model = KerasClassifier(build_fn=build_model, epochs=30, verbose=2)

cv = KFold(n_splits=3, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)
print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std()))
