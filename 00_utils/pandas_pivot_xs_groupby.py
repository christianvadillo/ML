# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:27:52 2020

@author: 1052668570
"""

import pandas as pd

# Data
train = pd.read_csv(r'../data/Regression/titanic_train.csv')
test = pd.read_csv(r'../data/Regression/titanic_test.csv')


train.head()
train.info()

# Groupby two classes and aggregate
grup1 = train.groupby(by=['Pclass', 'Survived'])['PassengerId'].count()

# Using pivot
pivot1 = train.pivot_table(values='PassengerId',
                           aggfunc='count',
                           columns='Survived',
                           index='Pclass')

print(grup1)
pivot1.xs(3)
print(pivot1)
pivot1.xs(3)

d = {'num_legs': [4, 4, 2, 2],
     'num_wings': [0, 0, 2, 2],
     'class': ['mammal', 'mammal', 'mammal', 'bird'],
     'animal': ['cat', 'dog', 'bat', 'penguin'],
     'locomotion': ['walks', 'walks', 'flies', 'walks']}
df = pd.DataFrame(data=d)
df = df.set_index(['class', 'animal', 'locomotion'])

print(df)

# Get values at specified index
df.xs('mammal')

# Get values at several indexes
df.xs(('mammal', 'dog'))

# Get values at specified index and level
df.xs('cat', level=1)  # level 1 == animal level
df.xs('walks', level=2)  # level 2 == locomotion level

# Get values at several indexes and levels
df.xs(('bird', 'walks'), level=[0, 'locomotion'])

# Get values at specified column and axis
df.xs('num_legs', axis=1)

