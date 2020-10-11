# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:05:00 2020

@author: 1052668570

https://www.kaggle.com/c/titanic/overview

On April 15, 1912, during her maiden voyage, the widely considered
“unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately,
there weren’t enough lifeboats for everyone onboard, resulting in the death of
 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving,
it seems some groups of people were more likely to survive than others.

We ask you to build a predictive model that answers the question: 
    “what sorts of people were more likely to survive?” 
using passenger data (ie name, age, gender, socio-economic class, etc).

The data has been split into two groups:
    training set (train.csv)
    test set (test.csv)

For the training set, we provide the outcome (also known as the “ground truth”)
for each passenger. Your model will be based on “features” like passengers’
gender and class. You can also use feature engineering to create new features.

For the test set, we do not provide the ground truth for each passenger.
It is your job to predict these outcomes. For each passenger in the test set,
use the model you trained to predict whether or not they survived the sinking
of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all
and only female passengers survive, as an example of what a submission file 
should look like.

survival	Survival	           0 = No, 1 = Yes
pclass	    Ticket class	       1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex
Age	        Age in years
sibsp	    # of siblings / spouses aboard the Titanic
parch	    # of parents / children aboard the Titanic
ticket	    Ticket number
fare	    Passenger fare
cabin	    Cabin number
embarked	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
"""

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import cufflinks as cf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")

# Data
train = pd.read_csv(r'../data/Regression/titanic_train.csv')
test = pd.read_csv(r'../data/Regression/titanic_test.csv')


# Exploring data
train.head(1).T
train.info()
# sns.pairplot(train, height=1.8, aspect=1, plot_kws={'s': 14, 'alpha': 0.3})
# How many people survived by sex?
ax = sns.countplot(train['Survived'],
                   hue=train['Sex'],
                   palette='RdBu_r')
ax.set_xticklabels(["No", "Si"])
plt.show()

# How many people survived by ticket class?
ax = sns.countplot(train['Survived'],
                   hue=train['Pclass'],
                   palette='YlGnBu')
ax.legend(["Upper", "Middle", "Lower"])
ax.set_xticklabels(["No", "Si"])
plt.show()

# age distribution
sns.distplot(train['Age'].dropna(), bins=30)

# How many people survived by ticket class?
ax = sns.countplot(train['Survived'],
                   hue=train['SibSp'],
                   palette='YlGnBu')
ax.set_xticklabels(["No", "Si"])
plt.show()

# Missing values?
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis") 

# Fillin Mising values for Age
sns.boxplot(x='Pclass', y='Age', data=train, palette='YlGnBu')
# We can see that the average age for each class is 37, 29, 24


def fill_missing_values(cols):
    """ Transform a missing value for corresponding age
    using pclass as pivot"""
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age

# Filling NaN's
train['Age'] = train[['Age', 'Pclass']].apply(fill_missing_values, axis=1)
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis") 
# Removing 'Cabin' column because we do not have information to figure out 
# the missing values
train.drop('Cabin', inplace=True, axis=1)
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis") 
train.dropna(inplace=True)  # Droping any other missing values


# Computing Dummies variables for Sex and Embarked
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1)

# Droping useless columns
train.drop(['PassengerId', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()


# =============================================================================
# Logistic Regression
# =============================================================================
X = train.iloc[:, 1:]
y = train.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

