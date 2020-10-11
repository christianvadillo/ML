# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:54:55 2020

@author: 1052668570
Building this graph

Constant
N1-
    -
      1
         -  Add
N2- - 2 - - N3 - - - -> 3
Constant
"""
import tensorflow as tf


n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

n3.eval
print(tf.keras.backend.eval(n3))
tf.compat.v1.get_default_graph()  # memory address of the graph created above

# Creating new graph using Graph()
g = tf.Graph()
print(g)  # New graph with new memory address

# Set new graph as default graph
graph_one = tf.compat.v1.get_default_graph()  # Saving the current graph
print(graph_one)  

graph_two = tf.Graph()  # Creating new graph
print(graph_two)  

# Set graph_two as default graph
with graph_two.as_default():
    print(graph_two is tf.compat.v1.get_default_graph())

print(graph_two is tf.compat.v1.get_default_graph())

