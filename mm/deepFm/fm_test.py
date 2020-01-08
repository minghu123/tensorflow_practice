import pandas as pd

import tensorflow as tf

import numpy as np

from scipy.sparse import csr

##　生成一个矩阵
file = open("train.libsvm", encoding='utf-8')
label = []
row = []
column = []
value = []
i = 0;
for line in file:
    strs = line.split(" ")
    label.append(int(strs[0]))
    for tmp in range(1, strs.__len__()):
        strs2 = strs[tmp].split(":")
        row.append(i);
        column.append(int(strs2[0]))
        value.append(float(strs2[1]))
    i += 1

martix = csr.csr_matrix((value, (row, column)), (i, max(column) + 1))

##：我去45491 个维度；
print(martix.shape)

##开始对数据进行训练
weights={};

weights['embeding']=tf.Variable(tf.random_normal(None, 40), dtype=tf.float32,name="embeding")

weights['']
