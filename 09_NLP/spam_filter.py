# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:41 2020

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

from nltk.corpus import stopwords

sns.set_style("whitegrid")
# nltk.download_shell()  # to download stopwords package

# data = [line.rstrip() for line in open('../data/nlp/smsspamcollection/SMSSpamCollection')]

# for num, message in enumerate(data):
#     print(message)

messages = pd.read_csv('../data/nlp/smsspamcollection/SMSSpamCollection', 
                       sep='\t',
                       names=['label', 'message'])

messages.describe()
messages.groupby(by='label').describe()

# Length of each message
messages['length'] = messages['message'].apply(len)

messages['length'].plot.hist(bins=50)

messages.hist(column='length', by='label', bins=60, figsize=(12, 4))
# Looks that length is a good feature to identify ham and spam messages

# =============================================================================
# Cleaning messages
# =============================================================================
# Removing puntuations and stopwords
def clean_word(message):
    # message = "Sample Message! Notice: it has punctuation"
    nopun = [char for char in message if char not in string.punctuation]
    nopun = "".join(nopun)
    return [word for word in nopun.split() if word.lower() not in stopwords.words('english')]

# messages['message'].head(5).apply(clean_word)

# =============================================================================
# Transforming words into vectors (VECTORIZATION)
# =============================================================================
bow_transformer = CountVectorizer(analyzer=clean_word).fit(messages['message'])
print(len(bow_transformer.vocabulary_))  # size of the bag of words

# Example
mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])
print(bow4)  # Unique words in the message 4
print(bow4.shape)
# Get the word that appear 2 times in message 4
bow_transformer.get_feature_names()[4068]
bow_transformer.get_feature_names()[9554]

# Transforming all messages
messages_bow = bow_transformer.transform(messages['message'])
print("Shape of Spare Matrix: ", messages_bow.shape)
print("Non-Zero Ocurrences", messages_bow.nnz)

# =============================================================================
# Getting Inverse Document Frequency 
# =============================================================================
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Example for message 4
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

# Check de inverse document frequency for one specific word (University)
tfidf_transformer.idf_[bow_transformer.vocabulary_['University']]

# Transforming all messages
messages_tfidf = tfidf_transformer.transform(messages_bow)


# =============================================================================
# Creating Naive Bayes model (MultinominalNB)
# =============================================================================
model = MultinomialNB().fit(messages_tfidf, messages['label'])

# =============================================================================
# Evaluating
# =============================================================================
predictions = model.predict(messages_tfidf)


# =============================================================================
# Using Pipline to sumarize all the previous phases with split data
# =============================================================================
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'],
                                                                messages['label'], 
                                                                test_size=0.3)

# =============================================================================
# Creating the pipiline
# It requires a list of tuples [('name_process', Process to execute)]
# =============================================================================
pipeline = Pipeline([('bow', CountVectorizer(analyzer=clean_word)),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())])

pipeline.fit(msg_train, label_train)  # Applying the pipline to the train data

# Evaluating model
predictions = pipeline.predict(msg_test)
print(confusion_matrix(label_test, predictions))
print(classification_report(label_test, predictions))


