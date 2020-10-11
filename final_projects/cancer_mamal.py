# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:32:39 2020

@author: 1052668570
Nominal: # No quantiative importance
Ordinal: Order imports

7. Attribute Information:
   1. BI-RADS assessment: 1 to 5 (ordinal)
   2. Age: patient's age in years (integer)
   3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal) 
   4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
   5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
   6. Severity: benign=0 or malignant=1 (binominal)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

sns.set_style("whitegrid")


features_names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
data = pd.read_csv('../data/classification/mammographic_masses.data.txt',
                   names=features_names, dtype=np.float32, na_values='?')

data.head()
data.info()

# sns.pairplot(data)
# sns.distplot(data['Age'], bins=20)
# sns.countplot(data['Severity'])
# sns.lmplot('Severity', 'Age', data=data)

# plt.figure()
# data[data['Severity'] == 0]['Age'].hist(bins=30, color='red',
#                                               label='Severity=Bening',
#                                               alpha=0.4)
# data[data['Severity'] == 1]['Age'].hist(bins=30, color='blue',
#                                               label='Severity=Malignant',
#                                               alpha=0.5)
# plt.legend()
# plt.xlabel("Age")

# =============================================================================
# Missing values?
# =============================================================================
data.isnull().sum() / len(data) * 100

# data.dropna(subset=['Age'], inplace=True)

# sns.distplot(data['BI-RADS'].dropna())
data['BI-RADS'].value_counts()
data['BI-RADS'][data['BI-RADS'] == 55] = 5.0
data['BI-RADS'][data['BI-RADS'] == 0.0] = 1.0

data[data['Shape'].isnull()]
data.groupby(['BI-RADS', 'Margin'])['Shape'].median()
data['Density'].value_counts()

data.dropna(inplace=True)

# =============================================================================
# Transforming categoricals (nominbals) to dummies [One-hot encoding]
# =============================================================================
data = pd.get_dummies(data=data, columns=['Shape', 'Margin'], drop_first=True, prefix=['Shape', 'Margin'])

# =============================================================================
# Splitting data
# =============================================================================
X = data.drop('Severity', axis=1).values
y = data['Severity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=101)

# =============================================================================
# Scaling data
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# Random forest
# =============================================================================
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cv_score_rf = cross_val_score(rf, X_train, y_train, cv=10)

print('cv:', cv_score_rf.mean())
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =============================================================================
# LogisticRegression
# =============================================================================
lr = LogisticRegression()

lr.fit(X_train, y_train)
cv_score_lr = cross_val_score(lr, X_train, y_train, cv=10)

print('cv:', cv_score_lr.mean())
y_pred_lr = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# =============================================================================
# SVM Model (Classification)
# =============================================================================
svc = SVC()
svc.fit(X_train, y_train)
cv_score_svc = cross_val_score(svc, X_train, y_train, cv=10)

print('cv:', cv_score_svc.mean())
y_pred_svc = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# Using GridSearch to find best hyperparameters
# param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, #First option - linear
#               {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}] # Second option - nonlinear

C_range = 10.0 ** np.arange(-4, 2)
coef0_range = 10.0 ** np.arange(-4, 2)
degree = np.arange(2, 6)
gamma_range = 10.0 ** np.arange(-4, 2)
# param_grid = [dict(C=C_range.tolist(), kernel=['linear'], class_weight=['balanced']),
#               dict(C=C_range.tolist(), kernel=['rbf'], gamma=gamma_range.tolist(),class_weight=['balanced']),
#               dict(C=C_range.tolist(), kernel=['sigmoid'], gamma=gamma_range.tolist(), coef0=coef0_range.tolist(),class_weight=['balanced']),
#               dict(C=C_range.tolist(), kernel=['poly'], gamma=gamma_range[:-2].tolist(), coef0=coef0_range.tolist(),class_weight=['balanced']),
#               ]

# Just Poly
param_grid = [dict(C=C_range.tolist(), 
                   kernel=['poly'], 
                   gamma=gamma_range[:-2].tolist(), 
                   coef0=coef0_range.tolist(),
                   class_weight=['balanced'],
                   degree=degree.tolist())]

grid = GridSearchCV(SVC(), param_grid,  refit=False, verbose=3)
grid.fit(X_train, y_train)

# grid.best_params_
# grid.best_estimator_
# grid.best_score_

# Evaluating grid model
svc2 = SVC(C=1.0,
           class_weight='balanced',
           coef0=10.0,
           degree=3,
           gamma=0.01,
           kernel='poly')
svc2.fit(X_train, y_train)
cv_score_svc2 = cross_val_score(svc2, X_train, y_train, cv=10)

print('cv:', cv_score_svc2.mean())
y_pred_svc2 = svc2.predict(X_test)
print(confusion_matrix(y_test, y_pred_svc2))
print(classification_report(y_test, y_pred_svc2))


# =============================================================================
# ANN
# =============================================================================
X_train.shape
ann = Sequential()
ann.add(Dense(16, activation='relu'))
ann.add(Dropout(0.5))
ann.add(Dense(8, activation='relu'))
ann.add(Dropout(0.5))
ann.add(Dense(8, activation='linear'))
ann.add(Dropout(0.5))
ann.add(Dense(8, activation='sigmoid'))
ann.add(Dropout(0.5))
ann.add(Dense(1, activation='sigmoid'))

ann.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'Recall'])

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
ann.fit(X_train, y_train,
        epochs=500,
        batch_size=64,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping])

# =============================================================================
# Evaluating model
# =============================================================================
loss_df = pd.DataFrame(ann.history.history)
loss_df.plot()

cv_score_ann = cross_val_score(ann, X_train, y_train, cv=10)

print('cv:', cv_score_ann.mean())
y_pred = ann.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))


def create_model():
    ann = Sequential()
    ann.add(Dense(32, activation='relu'))
    ann.add(Dropout(0.5))
    ann.add(Dense(16, activation='linear'))
    ann.add(Dropout(0.5))
    ann.add(Dense(16, activation='relu'))
    ann.add(Dropout(0.5))
    ann.add(Dense(16, activation='linear'))
    ann.add(Dropout(0.5))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy', 'Recall'])

    return ann


estimator = KerasClassifier(build_fn=create_model, epochs=500,
                            batch_size=64, verbose=2)

cv_scores = cross_val_score(estimator, X_train, y_train, cv=5)

cv_scores.mean()