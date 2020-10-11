# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:20:27 2020

@author: 1052668570
"""
from fastai import datasets
import pickle
import gzip
from torch import tensor


def get_data():
    MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl'
    path = datasets.download_data(MNIST_URL, ext='.gz',
                                  fname='mnist.gz')

    with gzip.open(path, 'rb') as f:
        ((X_train, y_train), (X_test, y_test), _) =\
            pickle.load(f, encoding='latin-1')
    # Transform to tensor
    print(path)
    return map(tensor, (X_train, y_train, X_test, y_test))


def normalize(x, mean, std):
    return (x - mean) / std