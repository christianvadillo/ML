# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:16:40 2020

@author: 1052668570

VISUALIZE THE ACTIVATION OF THE INNER LAYER

Used to understand what our model is doing and for dimensional reduction of
the dataset
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import keras.backend as K

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

sns.set_style('whitegrid')

data = pd.read_csv('../data/classification/banknotes.csv')

data.info()
data.describe()
data['class'].value_counts()

# sns.pairplot(data, hue='class')

# =============================================================================
# Spliting data
# =============================================================================
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# =============================================================================
# VISUALIZING INNER LAYERS
# =============================================================================
K.clear_session()

model = Sequential()
model.add(Dense(4, input_shape=(4,), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

h = model.fit(X_train, y_train,
              batch_size=16, epochs=20,
              verbose=1, validation_data=(X_test, y_test))

result = model.evaluate(X_test, y_test)
print(result)


model.summary()
model.layers[0]

# =============================================================================
# Getting the inputs and outputs of first inner layer
# =============================================================================
inp = model.layers[0].input  # 4 inputs (All features of dataset)
out = model.layers[1].output  # The 2 outpus from the relu act func

# Extracting output of a intermediate layer(first)
""" this function runs the computation graph that we have created in the code,
     taking input from the first parameter and extracting the number of outputs
     as per the layers mentioned in the second parameter."""
features_functions = K.function([inp], [out])
features = features_functions([X_test])[0]  # output of first layer (4 nodes)
features.shape

# Visualizing 
plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
""" We can see how our first inner layer is learning to separate
    the bank notes"""

# ===========================================================================
# =============================================================================
# Visualiazing how the model learn in each epoch
# =============================================================================

K.clear_session()

model = Sequential()
model.add(Dense(16, input_shape=(4,), activation='tanh'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

inp = model.layers[0].input  # 4 inputs (All features of dataset)
out = model.layers[2].output  # The 2 outpus from the relu act func

# Extracting output of a intermediate layer(first)
""" this function runs the computation graph that we have created in the code,
     taking input from the first parameter and extracting the number of outputs
     as per the layers mentioned in the second parameter."""
features_functions = K.function([inp], [out])

for i in range(1, 17):
    plt.subplot(4, 4, i)
    h = model.fit(X_train,  y_train, batch_size=16, epochs=1, verbose=0)
    test_accuracy = model.evaluate(X_test, y_test)[1]
    features = features_functions([X_test])[0]
    plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
    # plt.xlim(-0.5, 3.5)
    # plt.ylim(-0.5, 4.0)
    plt.title(f'Epoch: {i}, Test Acc: {test_accuracy * 100}')
plt.tight_layout()