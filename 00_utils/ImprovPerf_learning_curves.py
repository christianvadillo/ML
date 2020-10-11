# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:10:30 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping



# Loading data
data = load_digits()
X, y = data.data, data.target
X.shape

# Visualizing
# for i in range(8):
#     # images of 8x8 pixels
#     plt.subplot(1, 8, i+1)
#     plt.imshow(X.reshape(-1, 8, 8)[i], cmap='gray')

# =============================================================================
# Transforming labels using one hot encoding
# =============================================================================
y_cat = to_categorical(y, num_classes=10)

# =============================================================================
# Splitting the data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=101)

# =============================================================================
# Create an array of incrementaly train sizes
# 10%, 40$, 70%, 99.99% of training data
# =============================================================================
train_sizes = (len(X_train) * np.linspace(0.1, 0.999, 4)).astype(int)

# =============================================================================
# Setting up the first model[Using Functional API]
# =============================================================================
K.clear_session()

# Input layer
inputs = Input(64, )
# First layer
x = Dense(16, activation='relu')(inputs)
# output layer
outputs = Dense(10, activation='softmax')(x)

# Registring the model
model = Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=1)
# =============================================================================
# Store the initial random weights
# =============================================================================
initial_weights = model.get_weights()


# =============================================================================
# Creating the learning curve with the train_sizes
# =============================================================================
train_scores = []
test_scores = []
for train_size in train_sizes:
    X_train_frac, _, y_train_frac, _ = \
        train_test_split(X_train, y_train, train_size=train_size)

# at each iteration reset the weights of the model to the
# initial random weights
    model.set_weights(initial_weights)

    # Fitting the model
    h = model.fit(X_train_frac, y_train_frac,
                  epochs=300,
                  verbose=0,
                  callbacks=[early_stopping])
    # Training scores
    r = model.evaluate(X_train_frac, y_train_frac, verbose=0)
    train_scores.append(r[-1])
    # Test scores
    e = model.evaluate(X_test, y_test, verbose=0)
    test_scores.append(e[-1])

    print("Done size: ", train_size)

# =============================================================================
# Plotting the learning curve
# =============================================================================
plt.plot(train_sizes, train_scores, 'x-', label='Training score')
plt.plot(train_sizes, test_scores, 'o-', label='Test score')
plt.legend(loc='best')
plt.title('Maybe we need change the model' \
          if test_scores[-1] < test_scores[-2] \
              else 'Maybe we need to add more data')

