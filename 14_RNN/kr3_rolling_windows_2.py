# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:23:52 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from keras.layers import Dense, LSTM
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
# Creating dataframes
# =============================================================================
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
train_sc_df.head()

# =============================================================================
# Building scale time-series
# We will shift the dataframe 12 periods (months) starting from 1 to 12
# Given a table like:
""" 
               Scaled   shift_1      shift_2
Adjustments                                                            
1991-01-31   0.014020       NaN     
1991-02-28   Actual month->0.000000  0.014020<-Previous month
1991-03-31   Actual month->0.070426  0.000000  0.014020<-Previous two months
1991-04-30   Actual month->0.095318  0.070426  0.000000 <-Previous three months
"""
# =============================================================================
for s in range(1, 13):
    train_sc_df[f'shift_{s}'] = train_sc_df['Scaled'].shift(s)
    test_sc_df[f'shift_{s}'] = test_sc_df['Scaled'].shift(s)

# train_sc_df.plot()

# =============================================================================
# Defininf our target (Y)
# =============================================================================

X_train = train_sc[:-1]  # All except the last one
y_train = train_sc[1:]  # All except the first one

X_test = test_sc[:-1]  # All except the last one
y_test = test_sc[1:]  # All except the first one

# =============================================================================
# Shapping data for LSTM
# =============================================================================
""" LSTM requires 3D Tensor with shape (batch_size, timesteps, input_dim) """
# since we only have 1 column, our input_dim will be 1
# for timesteps is also be 1
X_train.shape  # Current shape is (239, 1)
X_train[:, None].shape  # Will add an empty dimension given us (239, 1, 1)

X_train_t = X_train[:, None]
X_test_t = X_test[:, None]


# =============================================================================
# Setting up the model [RNN- LSTM]
# =============================================================================
K.clear_session()

model = Sequential()
model.add(LSTM(units=6, input_shape=(1, 1)))  #( 1 timestep, 1 number)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# =============================================================================
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

model.fit(X_train_t, y_train, epochs=200, batch_size=1,
          callbacks=[early_stopping])

y_pred = model.predict(X_test_t)

plt.plot(test_sc)
plt.plot(y_pred)
