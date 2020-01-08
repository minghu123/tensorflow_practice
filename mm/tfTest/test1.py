import tensorflow as tf
import numpy as np

"""
测试不同维度的相乘;
"""

a = np.arange(0, 16, step=1, dtype=np.int).reshape((8, 2))

b = np.arange(0, 24, step=1, dtype=np.int).reshape((12, 2))

a_ = tf.placeholder(shape=(None, 2), dtype=tf.int32)

b_ = tf.placeholder(shape=(None, 2), dtype=tf.int32)

mat = tf.multiply(a_, b_)
inter = tf.concat((a_, b_), axis=-1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(mat, feed_dict={a_: a, b_: b}))
    print(sess.run(inter, feed_dict={a_: a, b_: b}))
