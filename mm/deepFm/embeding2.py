
import tensorflow as tf
import numpy as np


def embedding_lookup1():
    a = np.arange(12).reshape(3, 4)
    b = np.arange(12, 16).reshape(1, 4)
    c = np.arange(16, 28).reshape(3, 4)
    print(a)
    print('\n')
    print(b)
    print('\n')
    print(c)
    print('\n')

    a = tf.Variable(a)
    b = tf.Variable(b)
    c = tf.Variable(c)

    t = tf.nn.embedding_lookup([a, b, c],
                               partition_strategy='mod', ids=[2, 1, 6, 1, 3, 5, 0])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    m = sess.run(t)
    print(m)




def embedding_lookup2():
    a = np.arange(12).reshape(3, 4)
    b = np.arange(12, 16).reshape(1, 4)
    c = np.arange(16, 28).reshape(3, 4)
    print(a)
    print('\n')
    print(b)
    print('\n')
    print(c)
    print('\n')

    a = tf.Variable(a)
    b = tf.Variable(b)
    c = tf.Variable(c)

    t = tf.nn.embedding_lookup([a, b, c],
                               partition_strategy='div', ids=[5, 1, 0, 3, 2, 6])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    m = sess.run(t)
    print(m)




embedding_lookup2()
