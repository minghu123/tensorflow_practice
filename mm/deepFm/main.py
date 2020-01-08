import pandas as pd
import tensorflow as tf
import numpy as np;

## 使用panda读取数据；

train = pd.read_csv("train.csv")

# 对数据进行处理；
a = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
a = np.asarray(a)
idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
idx2 = tf.Variable([[0, 2, 3, 1], [4, 0, 2, 2]], tf.int32)
out1 = tf.nn.embedding_lookup(a, idx1)
out2 = tf.nn.embedding_lookup(a, idx2)
init = tf.global_variables_initializer()
"""
[[0.1 0.2 0.3]
 [2.1 2.2 2.3]
 [3.1 3.2 3.3]
 [1.1 1.2 1.3]]

原先的矩阵是5 *3 矩阵，然后embedding [0,2,3,1]，就是依次选取原来矩阵的 第0行，第2行，第3行，第1行组成一个新的矩阵；
5*3 =4*3 也就是 4*5 矩阵 来乘以这个矩阵
1  0  0  0  0           0.1, 0.2, 0.3       0.1 0.2 0.3
0  0  1  0  0   mat     0.1, 0.2, 0.3       2.1 2.2 2.3
0  0  0  1  0           2.1, 2.2, 2.3    =  3.1 3.2 3.3
0  1  0  0  0           3.1, 3.2, 3.3       1.1 1.2 1.3
                        4.1, 4.2, 4.3
"""
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(out1))
    print(sess.run(out2))

