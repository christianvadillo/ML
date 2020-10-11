# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:11:22 2020

@author: 1052668570
"""

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords


sns.set_style("whitegrid")


data = pd.read_csv('../data/nlp/yelp.csv')
# =============================================================================
# # Exploratory
# =============================================================================
data.head(1).T
data.info()
data.describe()

# =============================================================================
# Creating the text length column
# =============================================================================
data['length'] = data['text'].apply(len)

# =============================================================================
# Visualizing data
# =============================================================================
g = sns.FacetGrid(data=data, col='stars')
g.map(plt.hist, 'length', bins=50)

sns.boxplot(x='stars', y='length', data=data)
sns.countplot(data['stars'])

# =============================================================================
# Group by stars
# =============================================================================
data.groupby(by='stars').mean()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.ylim(0,5)

# =============================================================================
# Grabbing only reviews that were either 1 or 5 stars
# =============================================================================
data_class = data[(data['stars'] == 1) | (data['stars'] == 5)]

# =============================================================================
# Extracting the features
# =============================================================================
X = data_class['text']
y = data_class['stars']

# =============================================================================
# Transforming words into vectors (VECTORIZATION)
# =============================================================================
X = CountVectorizer().fit_transform(X)

# =============================================================================
# Splitting the data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# =============================================================================
# Training the model
# =============================================================================
nb = MultinomialNB()
nb.fit(X_train, y_train)

# =============================================================================
# Evaluating the model
# =============================================================================
y_pred = nb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# =============================================================================
# =============================================================================
# Training the model USING NLP and Pipline
# =============================================================================
def clean_word(message):
    # message = "Sample Message! Notice: it has punctuation"
    nopun = [char for char in message if char not in string.punctuation]
    nopun = "".join(nopun)
    return [word for word in nopun.split() if word.lower() not in stopwords.words('english')]


pipline = Pipeline([('get_bow', CountVectorizer(analyzer=clean_word)),
                    ('get_tfidf', TfidfTransformer()), # integer counts 
                    ('classify', RandomForestClassifier(n_estimators=100))
                    ])

X = data_class['text']
y = data_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipline.fit(X_train, y_train)

# =============================================================================
# Evaluating the model
y_pred = pipline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
