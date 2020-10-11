# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:40:48 2020

@author: 1052668570
"""

import tensorflow as tf


tensor1 = tf.keras.backend.random_uniform((4, 4), 0, 1)
tensor1

var1 = tf.Variable(initial_value=tensor1)


# Placeholders
""" 
Tf v 2.0

Placeholders don't quite make sense with eager execution since placeholders 
are meant to be "fed" in a call to Session.run, as part of the feed_dict 
argument. Since eager execution means immediate (and not deferred execution),
there is no notion of a session or a placeholder
 
import tensorflow as tf

def my_model(x):
  return tf.square(x) # you'd likely have something more sophisticated

x = tf.placeholder(tf.float32)
y = my_model(x)

with tf.Session() as sess:
  print(sess.run(y, feed_dict={x: 3.0})
        
With eager execution enabled, you can use my_model (and step through it,
debug it etc.) with much less boilerplate and no placeholders / sessions:

import tensorflow as tf

tf.enable_eager_execution()

def my_model(x):
  return tf.square(x)  # you'd likely have something more sophisticated

print(my_model(3.0)) 
"""
 