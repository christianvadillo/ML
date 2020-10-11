# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:13:39 2020

@author: 1052668570
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

data = pd.read_csv(r'../data/utils/international-airline-passengers.csv')

data.head()
data.info()

# Transforming string to date object
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

data.plot()
