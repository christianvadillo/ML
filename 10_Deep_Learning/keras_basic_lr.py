# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:05:25 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
sns.set_style("whitegrid")

data = pd.read_csv(r"../Tenserflow/DATA/fake_reg.csv")

data.head()

X = data[['feature1', 'feature2']].values
y = data['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# help(Sequential)
# =============================================================================
# ANN with 3 hidden layers for regression problem
# =============================================================================
"""
    Dense <-- Layer
    Dense(units=4, activation='relu') <-- Layer with 4 neurons. Every neuron
                                        is connected to every other neuron and
                                        using relu activation function
   model.compile(optimizer='', loss='') 
       -optimizer: is essentially how do you actually want to perform this gradient descent
       -loss: is going to change dependending on what we are trying to accomplish,
               a multi-class classification problem, binary classification or
               regression problem. [Check documentation help()]
"""
model = Sequential()  # Initilize the network
model.add(Dense(units=4, activation='relu'))  # Adding first layer with 4 neurons
model.add(Dense(units=4, activation='relu'))  # Adding second layer with 4 neurons
model.add(Dense(units=4, activation='relu'))  # Adding third layer with 4 neurons
model.add(Dense(units=1))  # Adding last layer (output)

model.compile(optimizer='rmsprop', loss='mse')  # For regression problem
# =============================================================================
# =============================================================================

model.fit(X_train, y_train, epochs=250)  # Training the model, 

# Loss decreased?
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# Evaluating data
model.evaluate(X_test, y_test, verbose=0)  # model loss on test data
model.evaluate(X_train, y_train, verbose=0)  # model loss on training data

y_pred = model.predict(X_test).flatten()
y_df = pd.DataFrame([y_test, y_pred]).transpose()
y_df.columns = ['true y', 'pred y']
y_df.plot()
y_df.plot.scatter('true y', 'pred y', alpha=0.3)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


# Residuals (Error)
y_df['Error'] = y_df['true y'] - y_df['pred y']
sns.distplot(y_df['Error'], bins=40)


# =============================================================================
# # New data
# =============================================================================
# [[Feature1, Feature2]]
new_gem = [[998, 1000]]
# Don't forget to scale!
new_gem = sc.transform(new_gem)
model.predict(new_gem)

# =============================================================================
# Saving model
# =============================================================================
model.save('models/my_model.h5')  # creates a HDF5 file 'my_model.h5'

# =============================================================================
# Loading model
# =============================================================================
loaded_model = load_model('models/my_model.h5')
loaded_model.predict(new_gem)
