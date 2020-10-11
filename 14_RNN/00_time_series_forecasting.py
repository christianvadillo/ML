# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:23:11 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import MonthEnd



data = pd.read_csv('../data/time_series/cansim-0800020-eng-6674700030567901031.csv',
                   skiprows=6, skipfooter=9,
                   engine='python')


# Transforming string date to datetime object and reformatting to
# show the last day of the month
data['Adjustments'] = pd.to_datetime(data['Adjustments']) + MonthEnd(1)

data.set_index([data['Adjustments']], inplace=True)
data.drop('Adjustments', axis=1, inplace=True)
data.head()
data.plot()


# =============================================================================
# Correctly way to split time series data
# =============================================================================
split_date = pd.Timestamp('01-01-2011')
train = data.loc[:split_date, ['Unadjusted']]
test = data.loc[split_date:, ['Unadjusted']]

# Visualizing
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

# =============================================================================
# Rescaling the data
# =============================================================================
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

# =============================================================================
# Defininf our target (Y)
# =============================================================================

X_train = train_sc[:-1]  # All except the last one
y_train = train_sc[1:]  # All except the first one

X_test = test_sc[:-1]  # All except the last one
y_test = test_sc[1:]  # All except the first one

# =============================================================================
# Setting up the model [Fully connected predictor]
# =============================================================================
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

early_stopping = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train, y_train, epochs=200, batch_size=2,
          callbacks=[early_stopping])

y_pred = model.predict(X_test)

plt.plot(test_sc)
plt.plot(y_pred)
