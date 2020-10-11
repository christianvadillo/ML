# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:29:35 2020

@author: 1052668570
"""

import tensorflow as tf
print(tf.__version__)


# Creating a constant tensor
hello = tf.constant("Hello ")
world = tf.constant("World")

type(hello)
type(world)
print(hello)

msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

a = tf.constant(10)
b = tf.constant(20)

print(tf.keras.backend.eval(a + b))


const = tf.constant(10)

fill_mat = tf.fill((4, 4), 10)  # Create a 4x4 matrix of all 10
fill_mat.eval
print(tf.keras.backend.eval(fill_mat))

myzeros = tf.zeros((4, 4))  # 4x4 matrix of zeros
myzeros.eval
print(tf.keras.backend.eval(myzeros))

ones = tf.ones((4, 4))  # 4x4 matrix of ones
ones.eval
print(tf.keras.backend.eval(ones))

# 4x4 random matrix with normal distribution
myrandn = tf.keras.backend.random_normal((4, 4), mean=0, stddev=1.0)
myrandn.eval
print(tf.keras.backend.eval(myrandn))

# 4x4 random matrix with uniform distribution
myrandu = tf.keras.backend.random_uniform((4, 4), minval=0, maxval=1)
myrandu.eval
print(tf.keras.backend.eval(myrandu))


## Executing all previous tensors
my_operations = [const, fill_mat, myzeros, ones, myrandn, myrandu]
print(tf.keras.backend.eval(my_operations))

for op in my_operations:
    print(op.eval)
    print()

# Matrices
mat1 = tf.constant([[1, 2],
                    [3, 4]])
mat1.shape

mat2 = tf.constant([[10], [200]])
mat2.shape

# Dot multiplication
result = tf.matmul(mat1, mat2)
result.eval
print(tf.keras.backend.eval(result))
