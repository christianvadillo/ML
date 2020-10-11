# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:12:00 2020

@author: 1052668570
Sentiment analysis from movie reviews
This notebook is inspired by the imdb_lstm.py example that ships with Keras.
But since I used to run IMDb's engineering department, I couldn't resist!

It's actually a great example of using RNN's. The data set we're using consists
of user-generated movie reviews and classification of whether the user liked 
the movie or not based on its associated rating.

More info on the dataset is here:

https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

So we are going to use an RNN to do sentiment analysis on full-text
movie reviews!

Think about how amazing this is. We're going to train an artificial neural
network how to "read" movie reviews and guess whether the author liked the
movie or not from them.

Since understanding written language requires keeping track of all the words
in a sentence, we need a recurrent neural network to keep a "memory" of the
words that have come before as it "reads" sentences over time.

In particular, we'll use LSTM (Long Short-Term Memory) cells because we don't
really want to "forget" words too quickly - words early on in a sentence can
affect the meaning of that sentence significantly.

y_train:
    They are just 0 or 1, which indicates whether the reviewer said they 
    liked the movie or not.

"""


# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb


print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

"""
So to recap, we have a bunch of movie reviews that have been converted into
vectors of words represented by integers, and a binary sentiment 
classification to learn from.
"""

# to keep things managable on our little PC let's limit the reviews to their first 80 words
X_train = sequence.pad_sequences(X_train, maxlen=80)
X_test = sequence.pad_sequences(X_test, maxlen=80)

# =============================================================================
# Setting up the RNN model
# =============================================================================

model = Sequential()
# start with an Embedding layer - this is just a step that converts the input 
# data into dense vectors of fixed size that's better suited for a neural network
# The 20,000 indicates the vocabulary size (remember we said we only wanted the top 20,000 words)
# and 128 is the output dimension of 128 units.
model.add(Embedding(20000, 128))
# set up a LSTM layer for the RNN itself.
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =============================================================================
# Training the model
#  Even with a GPU, this will take a long time!
# =============================================================================
model.fit(X_train, y_train,
          batch_size=32,
          epochs=2,
          verbose=2,
          validation_data=(X_test, y_test))

loss_df = pd.DataFrame(model.history.history)
loss_df[['loss', 'val_loss']].plot()

# =============================================================================
# Evaluating
# =============================================================================
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
