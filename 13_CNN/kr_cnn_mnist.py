# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:58:26 2020

@author: 1052668570

Mnist data using CNN and keras
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist

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

"""
We need to shape the data differently then before. Since we're treating the
data as 2D images of 28x28 pixels instead of a flattened stream of 784 pixels,
we need to shape it accordingly. Depending on the data format Keras is set up
for, this may be 1x28x28 or 28x28x1 (the "1" indicates a single color channel,
as this is just grayscale. If we were dealing with color images,
it would be 3 instead of 1 since we'd have red, green, and blue color channels)
"""

# Reshape
if K.image_data_format() == 'channel_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)

else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

X_train.shape
# Setting type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Scaling
X_train /= 255.
X_test /= 255.

# =============================================================================
# Encoding labels
# =============================================================================
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# display_sample(33)

# =============================================================================
# Setting up the model
# =============================================================================
model = Sequential()

# starting with a 2D convolution of the image
# it's set up to take 32 windows, or "filters", of each image, each filter being 
# 3x3 in size.
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# We then run a second convolution on top of that with 64 3x3 windows
model.add(Conv2D(64, (3, 3), activation='relu'))  # 64 3x3 kernels
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =============================================================================
# Training the model
""" This will be slow without GPU - Try running on google colab"""
# =============================================================================
model.fit(X_train, y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, y_test))

loss_df = pd.DataFrame(model.history.history)
loss_df[['loss', 'val_loss']].plot()

# =============================================================================
# Evaluating
# =============================================================================
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
