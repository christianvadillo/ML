# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:24:25 2020

@author: 1052668570
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


def display_sample(num):
    # Print sample label
    label = y_train[num]
    print(f'one hot array: {label}')

    # Transforming back to a number
    label = label.argmax(axis=0)
    # Reshape the 784 values to a 28*28 image
    image = X_train[num].reshape([28, 28])
    plt.title('Sample: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))


# MNIST dataset parameters
num_classes = 10  # total classes (0-10 digits)
num_features = 784  # data features (img shape: 28 * 28)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Converting to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Flatten images to 1-D vector of 784 features
# X_train2 = X_train.reshape([-1, num_features])
X_train = X_train.reshape(60000, 784)  # Same as above ^
X_test = X_test.reshape([-1, num_features])

# Normalize images values from [0, 255] to [0, 1]
X_train = X_train / 255.
X_test = X_test / 255.

# =============================================================================
# One hot encoding labels
# =============================================================================
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# Visualizing
# display_sample(2110)
# # How data is being feed into the network
# display_data(2110)


# =============================================================================
# Setting up the model (1-layer)
# =============================================================================
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784, )))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', input_shape=(784, )))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', input_shape=(784, )))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Model Summary
# model.summary()

# =============================================================================
# Setting up the optimizer
# =============================================================================
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =============================================================================
# Training the model
# =============================================================================
history = model.fit(X_train, y_train,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_test, y_test))

# =============================================================================
# Evaluating
# =============================================================================
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# =============================================================================
# Loss 
# =============================================================================
df_loss = pd.DataFrame(model.history.history)
df_loss[['loss', 'val_loss']].plot()
