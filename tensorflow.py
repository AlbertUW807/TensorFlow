# Libraries
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

# Loss of one training set
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss

# Create Tensors
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)

# Create and run session
sess = tf.Session()
print(sess.run(c))

x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

# Coding up the Linear Function
def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()

    return result

# Sigmoid function helper
def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result
