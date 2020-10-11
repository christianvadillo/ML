# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:53:23 2020

@author: 1052668570
Keras functional API. So far we've always used the Sequential model API in Keras
However, Keras also offers a Functional API, which is much more powerful.
You can find its documentation here. Let's see how we can leverage it.

define an input layer called inputs
define two hidden layers as before, one with 8 nodes, one with 5 nodes
define a second_to_last layer with 2 nodes
define an output layer with 3 nodes
create a model that connect input and output
train it and make sure that it converges
define a function between inputs and second_to_last layer
recalculate the features and plot them
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam

sns.set_style('whitegrid')
data = pd.read_csv('../data/classification/wines.csv')

data.info()
data.head(1).T

# =============================================================================
# Exploratory
# =============================================================================
data['Class'].value_counts()  # Multiclass mutually exclusives

# sns.pairplot(data, hue='Class')
plt.figure(figsize=(12, 7))
for i, n in enumerate(data.columns):
    plt.subplot(3, 6, i+1)
    plt.hist(data[n], bins=35)
    plt.title(n)

plt.tight_layout()


# =============================================================================
# Splitting data
# =============================================================================
X = data.iloc[:, 1:].values
y = to_categorical(data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Scaling data
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train, y_train)
X_test = sc.transform(X_test)

# =============================================================================
# Setting up the model using functional API
# =============================================================================

# Input layer
inputs = Input(shape=(13,))
# First layer
x = Dense(8, kernel_initializer='he_normal', activation='tanh')(inputs)
# Second layer
x = Dense(5, kernel_initializer='he_normal', activation='tanh')(x)
# second to last layer
second_to_last = Dense(2, kernel_initializer='he_normal', activation='tanh')(x)
# Output layer
outputs = Dense(4, activation='softmax')(second_to_last)

# Registring the model
model = Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(optimizer=Adam(0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)


# =============================================================================
# Exploring the layer with the 2 nodes
# =============================================================================
features_func = K.function([inputs], [second_to_last])
features = features_func(X_test)[0]

plt.scatter(features[:, 0], features[:, 1], c=np.argmax(y_test, axis=1), cmap='coolwarm')
