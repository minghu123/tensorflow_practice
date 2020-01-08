import pandas as pd

import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt

from scipy.sparse import csr

cols = ['user', 'item', 'rating', 'timestamp']

data = pd.read_csv('data/ua.base', delimiter="\t", names=cols)

user = data['user']
item = data['item']
rating = data['rating']
## 对用户进行编码
def getDict(unique):
    res = dict();
    i = 0
    for temp in unique:
        res[temp] = i;
        i += 1;
    return res;


userDict = getDict(user.unique());
itemDict = getDict(item.unique())
user1 = user.map(userDict)
item1 = item.map(itemDict)

mat = csr.csr_matrix((rating.values, (user1.values, item1.values)), shape=(len(userDict), len(itemDict)))
##构建矩阵

mat2 = mat.todense();
u = len(userDict)
n = len(itemDict)
userItem = tf.cast(tf.constant(mat2, shape=(u, n), dtype=tf.int64), dtype=tf.float64)

k = 40  ##确定向量的维度；
userF = tf.Variable(tf.random_normal((u, k), mean=0, stddev=1, dtype=tf.float64), name="userF", dtype=tf.float64)
itemF = tf.Variable(tf.random_normal((k, n), mean=0, stddev=1, dtype=tf.float64), name="itemF", dtype=tf.float64)
loss = tf.reduce_sum(tf.square(userItem - tf.matmul(userF, itemF))) + 0.1 * tf.reduce_sum(
    userF * userF) + 0.1 * tf.reduce_sum(itemF * itemF)

optimizer1 = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, var_list=[itemF])
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, var_list=[userF])
init = tf.global_variables_initializer();
y = []
x = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(600):
        sess.run(optimizer1)
        sess.run(optimizer2)
        temp = sess.run(loss)
        print(temp)
        x.append(i)
        y.append(temp)

plt.plot(x, y)
plt.show()
