# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:10:13 2020

@author: 1052668570

They've seen a lot of people leave the company recently and they would like
to understand why that's happening. They have collected historical data on
 employees and they would like you to build a model that is able to predict
 which employee will leave next. They would like a model that is better than
 random guessing. They also prefer false negatives than false positives, in
 this first phase. Fields in the dataset include:

Employee satisfaction level
Last evaluation
Number of projects
Average monthly hours
Time spent at the company
Whether they have had a work accident
Whether they have had a promotion in the last 5 years
Department
Salary
Whether the employee has left

Your goal is to predict the binary outcome variable left using the rest of
the data. Since the outcome is binary, this is a classification problem. Here
 are some things you may want to try out:

Establish a benchmark: what would be your accuracy score if you
predicted everyone stay?

Check if any feature needs rescaling. You may plot a histogram of the feature
 to decide which rescaling method is more appropriate.

convert the categorical features into binary dummy columns.
You will then have to combine them with the numerical features using pd.concat.

check the confusion matrix, precision and recall
check if you still get the same results if you use a 5-Fold cross validation
on all the data

Is the model good enough for your boss?
As you will see in this exercise, the a logistic regression model is not good
enough to help your boss. In the next chapter we will learn
how to go beyond linear models.

This dataset comes from https://www.kaggle.com/ludobenistant/hr-analytics/
and is released under CC BY-SA 4.0 License.

['satisfaction_level', 
 'last_evaluation', 
 'number_project',
 'average_montly_hours', 
 'time_spend_company', 
 'Work_accident',
  'left', 
  'promotion_last_5years', 
  'sales', 
  'salary']

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping


sns.set_style('whitegrid')
data = pd.read_csv("../data/classification/HR_comma_sep.csv")

data.head()
data.info()

# =============================================================================
# Establish a bechmark: What would be your accuracy score if you predicted
#
# =============================================================================
data['left'].value_counts() / len(data)
# Our model have to predict above 76% of employees leaving their job to make 
# a useful model

# =============================================================================
# Exploratory
# =============================================================================
stats = data.describe()

plt.figure(figsize=(15, 5))
for i, feature in enumerate(data.iloc[:,:-2].columns):
    plt.subplot(2, 4, i+1)
    data[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)

# =============================================================================
# Missing data?
# =============================================================================
data.isnull().sum()

# =============================================================================
# Transforming categorical
# =============================================================================
data['sales'].value_counts()
data = pd.get_dummies(data=data, drop_first=True, columns=['sales'])
data = pd.get_dummies(data=data, drop_first=True, 
                         columns=['number_project','time_spend_company'])


data['salary'].value_counts()
data['salary'] = data['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# =============================================================================
# Spliting data
# =============================================================================
X = data.drop('left', axis=1).values
y = data['left'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# =============================================================================
# Setting up the model
# =============================================================================
X_train.shape
model = Sequential()
model.add(Dense(1, input_shape=(27,), activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

# =============================================================================
# Evaluating model
# =============================================================================
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

y_pred = model.predict_classes(X_test).ravel()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# =============================================================================
# Validating with KFold Cross Validation
# =============================================================================
def build_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(27,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=build_model, epochs=20, batch_size=8)

score = cross_val_score(model, X_train, y_train, cv=cv)
print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(score.mean(), score.std()))
