# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:35:41 2020

@author: 1052668570

The Pima Indians dataset is a very famous dataset distributed by UCI and
originally collected from the National Institute of Diabetes and Digestive
and Kidney Diseases. It contains data from clinical exams for women age 21
and above of Pima indian origins. The objective is to predict based on
diagnostic measurements whether a patient has diabetes.

It has the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
The last colum is the outcome, and it is a binary variable.

In this first exercise we will explore it through the following steps:

For each feature draw a histogram.
    Bonus points if you draw all the histograms in the same figure.

Explore correlations of features with the outcome column.
You can do this in several ways, for example using the sns.pairplot
we used above or drawing a heatmap of the correlations.

Do features need standardization? If so what stardardization technique will
you use? MinMax? Standard?

Prepare your final X and y variables to be used by a ML model.
Make sure you define your target variable well. Will you need dummy columns?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import Recall, AUC
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

sns.set_style('whitegrid')

data = pd.read_csv('../data/classification/diabetes.csv')

# =============================================================================
# Exploratory
# =============================================================================
data.info()
stats = data.describe()

plt.figure(figsize=(10,9))
for n, column in enumerate(data.columns):
    plt.subplot(3, 3, n+1)
    plt.hist(data[column], bins=20)
    plt.title(column)

# =============================================================================
# Correlation?
# =============================================================================
sns.heatmap(data.corr(), cmap='viridis', annot=True)
plt.ylim(0, data.shape[1])

data.columns.ravel()
data['Pregnancies'].value_counts()
data.query('Pregnancies == 14')
sns.jointplot('Age', 'Pregnancies', data=data)

data['Glucose'].plot(kind='hist', bins=40)

sns.boxplot(data=data)


# =============================================================================
# Splitting data
# =============================================================================
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# Setting up the model
# =============================================================================
model = Sequential()
model.add(Dense(16, input_shape=(8,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='softplus'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[Recall()])

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_recall_25', mode='max', verbose=1, patience=10)

model.fit(X_train, y_train,
          batch_size=8, epochs=5000, 
          verbose=2, validation_data=(X_test, y_test))

# =============================================================================
# Evaluating model
# =============================================================================
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

y_pred = model.predict_classes(X_test).ravel()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
