# -*- coding: utf-8 -*-
"""kr_CNN_Cifar10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Frtb3hwVUdxlgdqz0GdUb-0u3y5V_2FC

### Cifra using CNN

Pleased with your performance with the digits recognition task, your boss decides to challenge you with a harder task. Their online branch allows people to upload images to a website that generates and prints a postcard that is shipped to destination. Your boss would like to know what images people are loading on the site in order to provide targeted advertising on the same page, so he asks you to build an image recognition system capable of recognizing a few objects. Luckily for you, there's a dataset ready made with a collection of labeled images. This is the [Cifar 10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html), a very famous dataset that contains images for 10 different categories:

- airplane 										
- automobile 										
- bird 										
- cat 										
- deer 										
- dog 										
- frog 										
- horse 										
- ship 										
- truck

In this exercise we will reach the limit of what you can achieve on your laptop and get ready for the next session on cloud GPUs.

Here's what you have to do:
- load the cifar10 dataset using `keras.datasets.cifar10.load_data()`
- display a few images, see how hard/easy it is for you to recognize an object with such low resolution
- check the shape of X_train, does it need reshape?
- check the scale of X_train, does it need rescaling?
- check the shape of y_train, does it need reshape?
- build a model with the following architecture, and choose the parameters and activation functions for each of the layers:
    - conv2d
    - conv2d
    - maxpool
    - conv2d
    - conv2d
    - maxpool
    - flatten
    - dense
    - output
- compile the model and check the number of parameters
- attempt to train the model with the optimizer of your choice. How fast does training proceed?
- If training is too slow (as expected) stop the execution and move to the next session!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import (Dense, Dropout, Conv2D, 
                                     MaxPooling2D, Flatten)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train[0]

X_train.shape

plt.imshow(X_train[3])

"""**Scaling the data**"""

X_train = (X_train / 255.).astype('float32')
X_test = (X_test / 255.).astype('float32')

"""Transforming Y labels"""

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""# **Setting up the CNN model**

* conv2d
*conv2d
*maxpool
*conv2d
*conv2d
*maxpool
*flatten
*dense
*output
"""

K.clear_session()
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), 
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='softplus', kernel_initializer='normal'))
model.add(Dropout(0.9))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.fit(X_train, y_train, batch_size=128, epochs=150,
          verbose=2, validation_data=(X_test, y_test))
          # callbacks=[early_stopping])

history_df = pd.DataFrame(model.history.history)
history_df.plot()



model.summary()

# model.fit(X_train, y_train, batch_size=8, epochs=100,
#           verbose=1, validation_data=(X_test, y_test),
#           callbacks=[early_stopping])

# history_df = pd.DataFrame(model.history.history)
# history_df.plot()

# history_df = pd.DataFrame(model.history.history)

# history_df.plot()