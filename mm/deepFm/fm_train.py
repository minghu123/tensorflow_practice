import numpy as np

import tensorflow as tf

import pandas as pd

"""
模拟Basic-DeepFM-model 的代码训练一个简单DeepFm 模型；
"""

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##对数据要进行处理，首先要去掉忽略的字段，还有对非数值型的字段进行oheHot 编码；


NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
    "missing_feat", "ps_car_13_x_ps_reg_03"
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]

cols = [col for col in train.columns if col not in IGNORE_COLS]

x_train = train[cols]
y_train = train['target']

## 对非数字的值进行oneHot编码，然后会形成一个稀疏矩阵
uc = dict();
dim = 0

for col in cols:
    if col in NUMERIC_COLS:
        uc[col] = dim
        dim += 1
        continue
    unique = x_train[col].unique()
    uc[col] = dict(zip(unique, range(dim, dim + len(unique))))
    dim += len(unique)
## 出现了这个dict字典；

x_index = np.ones_like(x_train.values, dtype=np.int)
x_value = np.ones_like(x_train.values, dtype=np.float)

for i, col in enumerate(cols):

    if type(uc[col]) == int:
        x_index[:, i] = uc[col];
        x_value[:, i] = x_train.values[:, i]
    else:
        x_index[:, i] = x_train[col].map(uc[col])
        x_value[:, i] = 1

print(x_index)
print(x_value)

filed_size = len(cols)
##　开始构建embeding 矩阵,先按照原始的论文来构建矩阵；
embeding_size = 20;

embeding = tf.Variable(tf.random_normal(shape=(dim + 1, embeding_size), dtype=tf.float64, mean=0, stddev=1.0),
                       name="embeding")

embeding_bias = tf.Variable(tf.random_normal(shape=(dim + 1, 1), dtype=tf.float64, mean=0, stddev=1.0),
                            name='embeding_bais');

t_x_index = tf.placeholder(dtype=tf.int32, shape=(None, filed_size), name="x_index")

t_x_value = tf.placeholder(dtype=tf.float64, shape=(None, filed_size), name="x_value")
y_label = tf.placeholder(dtype=tf.int32, shape=(None, 1), name="y_label")

y_label2 = tf.cast(y_label, dtype=tf.float64)

deep_embeding = tf.nn.embedding_lookup(embeding, t_x_index);  # N F k

feat_value = tf.reshape(t_x_value, shape=(-1, len(cols), 1))
deep_embeding = tf.multiply(deep_embeding, feat_value)

deep = tf.reduce_sum(deep_embeding, axis=1);

deep_wiegt0 = tf.Variable(tf.random_normal(shape=(embeding_size, 32), dtype=tf.float64, mean=0, stddev=1),
                          name='weight_0')
deep_wiegt1 = tf.Variable(tf.random_normal(shape=(32, 32), dtype=tf.float64, mean=0, stddev=1),
                          name='weight_1')

deep = tf.nn.relu(tf.matmul(deep, deep_wiegt0)) ## 这里没有设置偏置矩阵
deep = tf.nn.relu(tf.matmul(deep, deep_wiegt1))

first_order = tf.nn.embedding_lookup(embeding_bias, t_x_index)
first_order = tf.reshape(first_order, shape=(-1, filed_size))
first_order = tf.multiply(t_x_value, first_order)
##　first_order = tf.reduce_sum(first_order, axis=1)
second_sum = tf.square(tf.reduce_sum(deep_embeding, 1))  ##

second_2 = tf.reduce_sum(tf.square(deep_embeding), 1)

second_order = 0.5 * tf.subtract(second_sum, second_2)

input = tf.concat([first_order, second_order, deep], axis=1)

concat_w = tf.Variable(tf.random_normal(shape=(filed_size + embeding_size + 32, 1), dtype=tf.float64))

concat_b = tf.Variable(tf.constant(0.01, dtype=tf.float64))

y = tf.add(tf.matmul(input, concat_w), concat_b)
y = tf.sigmoid(y)

loss = tf.nn.l2_loss(tf.subtract(y,y_label2))
reglu = tf.contrib.layers.l2_regularizer(0.01)(deep_wiegt0) + tf.contrib.layers.l2_regularizer(0.01)(
    deep_wiegt1) + tf.contrib.layers.l2_regularizer(0.01)(concat_w) + tf.contrib.layers.l2_regularizer(0.01)(concat_b)

loss = tf.cast(loss, dtype=tf.float64)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,
                                   epsilon=1e-8).minimize(loss)

init = tf.global_variables_initializer();

with tf.Session() as sess:
    sess.run(init)
    print(reglu)
    print(loss)
    for i in range(0, 100):
        sess.run(optimizer,
                 feed_dict={t_x_index: x_index, t_x_value: x_value, y_label: y_train.values.reshape(-1, 1)})
        if i % 10 == 0:
            print(sess.run(loss,
                           feed_dict={t_x_index: x_index, t_x_value: x_value, y_label: y_train.values.reshape(-1, 1)}))

    count = 0
    y_pred = sess.run(y, feed_dict={t_x_index: x_index, t_x_value: x_value, y_label: y_train.values.reshape(-1, 1)})
    y_true = y_train.values
    for i, y_p in enumerate(y_pred):
        y_label = y_true[i]
        if y_p < 0.5 and y_label == 0:
            count += 1
        if y_p > 0.5 and y_label == 1:
            count += 1

    print(count)
