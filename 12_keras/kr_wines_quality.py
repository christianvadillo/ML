# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:09:10 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, Adagrad, RMSprop


sns.set_style('whitegrid')
data = pd.read_csv('../data/classification/wines.csv')

data.info()
data.head(1).T

# =============================================================================
# Exploratory
# =============================================================================
data['Class'].value_counts()  # Multiclass mutually exclusives

# sns.pairplot(data, hue='Class')
plt.figure(figsize=(12, 7))
for i, n in enumerate(data.columns):
    plt.subplot(3, 6, i+1)
    plt.hist(data[n], bins=35)
    plt.title(n)

plt.tight_layout()


# =============================================================================
# Splitting data
# =============================================================================
X = data.iloc[:, 1:].values
y = to_categorical(data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Scaling data
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train, y_train)
X_test = sc.transform(X_test)



# =============================================================================
# TESTING Hyperparameters
# =============================================================================
dflist = []

initializers = ['uniform', 'normal']


optimizers = ['Adam(lr=0.05)',
              'RMSprop(lr=0.01)',
              'RMSprop(lr=0.05)']

for opt in optimizers:
    K.clear_session()
    for init in initializers:
        K.clear_session()

        model = Sequential()
        model.add(Dense(8, input_shape=(13,), activation='relu', kernel_initializer=init))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=eval(opt),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        h = model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=0)

        dflist.append(pd.DataFrame(h.history, index=h.epoch))

historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([optimizers, initializers, metrics_reported],
                                 names=['optimizer', 'initializer', 'metric'])
historydf.columns = idx

# Plotting
plt.figure()
historydf.xs('loss', axis=1, level='metric').plot()
plt.title('Loss')
plt.xlabel("Epochs")
plt.show()

plt.figure()
historydf.xs('accuracy', axis=1, level='metric').plot()
plt.title('Accuracy')
plt.xlabel("Epochs")
plt.show()

# =============================================================================
# Setting up the model With Adam(0.5) - normal
# =============================================================================
X_train.shape

model = Sequential()
model.add(Dense(8, input_shape=(13,), activation='relu', kernel_initializer='normal'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=Adam(0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# =============================================================================
# Model performance
# =============================================================================
history_df = pd.DataFrame(model.history.history)
history_df.plot()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # same as y_pred = model.predict_classes(X_test)
y_test_classes = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_classes, y_pred))
print(classification_report(y_test_classes, y_pred))


# =============================================================================
# Exploring the layer with the 2 nodes
# =============================================================================
inp = model.layers[0].input
out = model.layers[2].output

features_func = K.function([inp], [out])
features = features_func(X_test)[0]

plt.scatter(features[:, 0], features[:, 1], c=y_test_classes, cmap='coolwarm')

# =============================================================================
# Validating with KFold Cross Validation
# =============================================================================
K.clear_session()


def build_model():
    model = Sequential()
    model.add(Dense(8, input_shape=(13,), activation='relu', kernel_initializer='normal'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=Adam(0.05),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


cv = KFold(n_splits=10)
model = KerasClassifier(build_fn=build_model, epochs=100, verbose=2)
scores = cross_val_score(model, X_train, y_train, cv=cv)

print(scores.mean())


