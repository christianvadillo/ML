# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:02:27 2020

@author: 1052668570
"""

import sys, wget, tarfile, shutil
import re
import random, math
from collections import Counter
import numpy as np


# =============================================================================
#         Get data
# =============================================================================
# url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz'
# file = wget.download(url)
# tar = tarfile.open('tasks_1-20_v1-1.tar.gz', "r:gz")
# tar.extractall('00_utils/grokking_book_scratch_codes')
# tar.close()

# =============================================================================
#  Load data
# =============================================================================
f = open('../grokking_book_scratch_codes/tasksv11/en/qa1_single-supporting-fact_train.txt', 'r')
raw = f.readlines()
f.close()

tokens = list()
rm_symbols = ['\n', '\t']

for line in raw:
    line = re.sub(r'[0-9]+', '', line)
    for s in rm_symbols:
        line = line.lower().replace(s, '')
    tokens.append(line.split(" ")[1:])


print(tokens[0:3])

# =============================================================================
# Setting things up
# =============================================================================
vocab = set()

for sent in tokens:
    for word in sent:
        vocab.add(re.sub(r'[0-9]+', '', word))


vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i
    
    
def word2indices(sentence):
    """ For plotting """
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# =============================================================================
# Initialize random seeds
# =============================================================================
np.random.seed(1)
embed_size = 10

# Word embeddings
embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1
# Inittially identity matrix
recurrent = np.eye(embed_size)
# Sentence embedding for an empty sentence
start = np.zeros(embed_size)
#Output weights
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1
# For the loss function
one_hot = np.eye(len(vocab))

# =============================================================================
# Forward propragation with arbitrary length
# =============================================================================
def predict(sent):
    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)
    
    loss = 0
    
    # Forward progragates
    preds = list()
    for target_idx in range(len(sent)):
        layer = {}
        # Tries to predict the next term
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))
        loss += -np.log(layer['pred'][sent[target_idx]])
        # Generates the next hidden state
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) +\
            embed[sent[target_idx]]
        layers.append(layer)
    return layers, loss


# =============================================================================
# Backward with arbitrary length
# =============================================================================
epochs= 30000
for epoch in range(epochs):
    alpha= 0.001
    sent = word2indices(tokens[epoch%len(tokens)][1:])
    # Forward
    layers, loss = predict(sent)

    # Backpropagates
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]
        
        # if not the first layer
        if(layer_idx > 0):
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose())
            
            if(layer_idx == len(layers) - 1):
                layer['hidden_delta'] = new_hidden_delta 
            else:
                layer['hidden_delta'] = new_hidden_delta + \
                    layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
        else:
            layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta']\
                .dot(recurrent.transpose())
                
    # Update weights
    start -= layers[0]['hidden_delta'] * alpha / float(len(sent))
    for layer_idx, layer in enumerate(layers[1:]):
        decoder -= np.outer(layers[layer_idx]['hidden'],\
                            layer['output_delta']) * alpha / float(len(sent))
            
        embed_idx = sent[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * \
            alpha / float(len(sent))
        recurrent -= np.outer(layers[layer_idx]['hidden'], \
                              layer['hidden_delta'] * alpha / float(len(sent)))
            
    if(epoch % 1000 == 0):
        print('Perplexity:' + str(np.exp(loss/len(sent))))


sent_index = 4
l, _ = predict(word2indices(tokens[sent_index]))
print(tokens[sent_index])

for i, each_layer in enumerate(l[1:-1]):
    input_ = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print("Prev input:"+ input_ + (' ' * (12 - len(input_))) +\
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)
 