import pandas as pd

import numpy as np
import tensorflow as tf

## 读取数据

data = pd.read_csv("train.csv")  ## 这里的正负样本分布非常不均衡；

##对非数值型数值进行oneHot 编码；

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

##怎么进行onehot编码；
idx = dict();  ## 这个dict的事元数据中的位置对应一个新的oheHot 中的位置，因为
dimlen = 0;
columnindex = -1
for col in data.columns:
    columnindex += 1
    if col in IGNORE_COLS:
        continue;
    if col in NUMERIC_COLS:
        idx[columnindex] = 1
        dimlen += 1;
    else:
        us = data[col].unique()
        idx[columnindex] = dict(zip(us, range(dimlen, dimlen + len(us))))
        dimlen += len(us);
print(idx)

datahot = np.zeros((data.shape[0], dimlen), dtype=float)

i = 0
for tmp in data.values.tolist():
    j = 0
    for temp2 in range(0, len(tmp)):
        if idx.get(temp2) == None:  ##需要丢弃的列
            continue;
        elif idx[temp2] == 1:  ##这里判断出其是数值型列
            datahot[i, j] = tmp[temp2]
            j += 1
        else:
            hotindex = idx[temp2][tmp[temp2]]
            datahot[i, hotindex] = 1
            j += len(idx[temp2])
    j = 0
    i += 1

print(datahot)

##对数据进行切分，切分为训练集和测试集；

train = datahot[0:8000, :]
y_tain = data['target'].values[0:8000]
test = datahot[8000:9999, :]
y_test = data['target'].values[8000:9999]

## 设置tf 的权重矩阵


w = tf.Variable(tf.random_normal((train.shape[1], 1), 0, 1), name="w")
b = tf.Variable(tf.constant(0.01), dtype=tf.float32, name='b')

x = tf.placeholder(tf.float32, (None, train.shape[1]), name='x')

y = tf.placeholder(tf.float32, (None, 1), name='y')

liner = tf.matmul(x, w) + b
sig = tf.nn.sigmoid(liner)
loss = tf.reduce_sum(tf.square(y - sig))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 100):
        sess.run(optimizer, feed_dict={x: train, y: y_tain.reshape(-1, 1)})
        print(sess.run(loss, feed_dict={x: train, y: y_tain.reshape(-1, 1)}))

    ## 计算预测的准确率；
    y_pred = sess.run(sig, feed_dict={x: train})
    clickNum = 0
    for i in range(0, 8000):
        tmp1 = y_pred[i][0]
        pred_label = 0
        if tmp1 > 0.5:
            pred_label = 1
        if pred_label == y_tain[i]:
            clickNum += 1
    print(clickNum)

    ##对测试数据集进行预测
    y_test_pred = sess.run(sig, feed_dict={x: test})
    print(type(y_test))

    clickNum2 = 0
    for i in range(0, 1999):
        tmp1 = y_test_pred[i][0]
        pred_label = 0
        if tmp1 > 0.5:
            pred_label = 1
        if pred_label == y_test[i]:
            clickNum2 += 1
    print(clickNum2)
