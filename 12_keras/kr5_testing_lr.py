# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:16:40 2020

@author: 1052668570
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import keras.backend as K

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam, Adagrad, RMSprop

sns.set_style('whitegrid')

data = pd.read_csv('../data/classification/banknotes.csv')

data.info()
data.describe()
data['class'].value_counts()

sns.pairplot(data, hue='class')

# =============================================================================
# Spliting data
# =============================================================================
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# =============================================================================
# Baseline model (Random Forrest)
# =============================================================================
rf = RandomForestClassifier()
cross_val_score(rf, X_train, y_train).mean()

# =============================================================================
# Logistic Regression Model (Keras)
# =============================================================================
model = Sequential()
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
result = model.evaluate(X_test, y_test)

history_df = pd.DataFrame(model.history.history)
history_df.plot()
plt.title('Test loss: {:3.5f}, Test accuracy: {:3.1f} %'.format(result[0], result[1]*100))

# =============================================================================
# TESTING LEARNING RATES
# =============================================================================
dflist = []
learning_rates = [0.01, 0.05, 0.1, 0.5]
""" 
Calling clear_session() releases the global graph state that Keras is holding
 on to; resets the counters used for naming layers and variables in Keras; 
 and resets the learning phase. This helps avoid clutter from old models 
 and layers, especially when memory is limited, and a common use-case for
 clear_session is releasing memory when building models and layers in a loop.
"""
for lr in learning_rates:
    K.clear_session()
    
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(optimizer=SGD(lr=lr), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, 
              epochs=10, batch_size=16, verbose=0)
    
    dflist.append(pd.DataFrame(model.history.history))

historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported],
                                 names=['learning_rate', 'metric'])
historydf.columns = idx

# Plotting
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('accuracy', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel("Epochs")
plt.tight_layout()

# =============================================================================
# TESTING BATCH SIZEs
# =============================================================================
dflist = []
batch_sizes = [16, 32, 64, 128]

for bs in batch_sizes:
    K.clear_session()
    
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(optimizer='sgd', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, 
              epochs=10, batch_size=bs, verbose=0)
    
    dflist.append(pd.DataFrame(model.history.history))

historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([batch_sizes, metrics_reported],
                                 names=['batch_size', 'metric'])
historydf.columns = idx

# Plotting
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1.1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('accuracy', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel("Epochs")
plt.tight_layout()


# =============================================================================
# TESTING OPTIMIZERS 
# =============================================================================
dflist = []
optimizers = ['SGD(lr=0.01)',
              'SGD(lr=0.01, momentum=0.3)',
              'SGD(lr=0.01, momentum=0.3, nesterov=True)',
              'Adam(lr=0.01)',
              'Adagrad(lr=0.01)',
              'RMSprop(lr=0.01)']

for opt in optimizers:
    K.clear_session()
    
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(optimizer=eval(opt), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    
    dflist.append(pd.DataFrame(model.history.history))

historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([optimizers, metrics_reported],
                                 names=['optimizer', 'metric'])
historydf.columns = idx

# Plotting
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1.1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('accuracy', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel("Epochs")
plt.tight_layout()


# =============================================================================
# TESTING INITIALIZERS 
# =============================================================================
dflist = []

initializers = ['zeros', 'uniform', 'normal',
                'he_normal', 'lecun_uniform']

for init in initializers:

    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,),
                    kernel_initializer=init,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    h = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
    
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([initializers, metrics_reported],
                                 names=['optimizer', 'metric'])
historydf.columns = idx

# Plotting
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1.1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('accuracy', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel("Epochs")
plt.tight_layout()