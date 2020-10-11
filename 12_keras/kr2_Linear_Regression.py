# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:15:05 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SDG


data = pd.read_csv('../data/Regression/weight-height.csv')

data.head()

data.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults')

X = data['Height'].values
y = data['Weight'].values

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.2)

# =============================================================================
# LR with keras
# =============================================================================
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(optimizer=Adam(lr=0.3), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, validation_data=((X_test, y_test)))

# =============================================================================
# Evaluating model
# =============================================================================
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

y_pred = model.predict(X_test).ravel()

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R^2: {r2_score(y_test, y_pred)}")


W, B = model.get_weights()



# model.summary()

