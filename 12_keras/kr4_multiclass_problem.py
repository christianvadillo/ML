# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:14:13 2020

@author: 1052668570
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


sns.set_style('whitegrid')
data = pd.read_csv("../data/classification/iris.csv")

# sns.pairplot(data, hue='species')
data.head()

data['species'].value_counts()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].map({name: i for i, name in
                          enumerate(data['species'].unique())}).values

# =============================================================================
# Creating dummy columns using keras
# =============================================================================
y_cat = to_categorical(y)


# =============================================================================
# Spliting data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y_cat,
                                                    test_size=0.2,
                                                    random_state=42)

# =============================================================================
# Setting up the model (Shallow model - Logistic regression)
# =============================================================================
model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(lr=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# validation_split = takes 10% of training data, leaves it out from the training
# and used for the testing loss
model.fit(X_train, y_train, epochs=20, validation_split=0.1)

y_pred = model.predict(X_test)

# Getting the class
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))
