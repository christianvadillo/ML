# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:37:19 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

data = pd.read_csv('../data/tesis/da_data_with_two_classes.csv', index_col=0)

data.head(1).T

data['Errors'].value_counts()
# sns.countplot(data['Errors'])

# Balancing data
sample = data.sample(int(len(data)*0.6))
sample = sample.groupby('Errors')
sample = sample.apply(lambda x: x.sample(sample.size().min()).reset_index(drop=True))
# sns.countplot(sample['Errors'])


# Scaling data
X = sample.iloc[:, :-1].values
y = sample.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=101, 
                                                    test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
# Training model
# =============================================================================
model = Sequential()
model.add(Dense(12, input_shape=(6,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(X_train, y_train,
          epochs=100, batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping], verbose=2)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

y_pred = model.predict_classes(sc.transform(data.iloc[:,:-1])).ravel()

print(classification_report(data.iloc[:,-1], y_pred))
print(confusion_matrix(data.iloc[:,-1], y_pred))

# =============================================================================
# Validating model
# =============================================================================
def build_model():
    model = Sequential()
    model.add(Dense(12, input_shape=(6,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=build_model,
                        epochs=100,
                        verbose=2,
                        batch_size=256)

cv = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X_train, y_train, cv=cv)
print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std()))



# =============================================================================
# Saving model
# =============================================================================
# model.save('models/da_ann_12_6_1.h5')  # creates a HDF5 file 'my_model.h5'
