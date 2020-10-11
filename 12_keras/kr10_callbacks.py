# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:18:22 2020

@author: 1052668570
Keras offers the possibility to call a function at each epoch. 
These are Callbacks, and their documentation is here. Callbacks allow us to add
 some neat functionality. In this exercise we'll explore a few of them.

Split the data into train and test sets with a test_size = 0.3 and 
random_state=42

Reset and recompile your model
train the model on the train data using validation_data=(X_test, y_test)
Use the EarlyStopping callback to stop your training if 
the val_loss doesn't improve
Use the ModelCheckpoint callback to save the trained model to disk once
 training is finished
 
 
 ######### TENSORBOARD ~~~~~~~~~~~~
Use the TensorBoard callback to output your training information to 
a /tmp/ subdirectory

1-create manually the directory
D:/PYTHON_PROJECTS/ML/TensorBoard/train/plugins


2-Then open a console.
3-Change directory to one that contain train folder
 i.e 
   cd  D:\PYTHON_PROJECTS\ML\12_keras\TensorBoard

4- run command:
    tensorboard --logdir=./
    
5- Open page

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K
import os

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

sns.set_style('whitegrid')

data = pd.read_csv('../data/classification/wines.csv')

data.info()
data.head(1).T

# =============================================================================
# Exploratory
# =============================================================================
data['Class'].value_counts()  # Multiclass mutually exclusives

# sns.pairplot(data, hue='Class')
# plt.figure(figsize=(12, 7))
# for i, n in enumerate(data.columns):
#     plt.subplot(3, 6, i+1)
#     plt.hist(data[n], bins=35)
#     plt.title(n)

# plt.tight_layout()


# =============================================================================
# Splitting data
# =============================================================================
X = data.iloc[:, 1:].values
y = to_categorical(data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# =============================================================================
# Scaling data
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train, y_train)
X_test = sc.transform(X_test)


# =============================================================================
# Callbacks
# =============================================================================
checkpointer = ModelCheckpoint(filepath='../12_keras/models/checkpointer.hdf5',
                               verbose=1,
                               save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=1, verbose=1, mode='auto')

logdir = os.path.join('TensorBoard')
tensorboard = TensorBoard(log_dir=logdir)
# =============================================================================
# Setting up the model using functional API
# =============================================================================
K.clear_session()
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
model.compile(optimizer=Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
model.fit(X_train, y_train, batch_size=16, epochs=20,
          verbose=1, validation_data=(X_test, y_test),
          callbacks=[checkpointer, early_stopping, tensorboard])

