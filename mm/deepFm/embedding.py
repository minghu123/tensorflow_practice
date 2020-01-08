import numpy as np

import tensorflow as tf;

"""
使用 embeding 来处理数据，
"""

example = np.arange(24).reshape(6, 4).astype(np.float32)
embedding = tf.Variable(example)

idx = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]],
                      values=[0, 0, 0, 0, 0], dense_shape=[3, 3])
"""
[[0 0 0]
 [0 0 0]  这里 [0 0 0 ] 代表用第一行来选取数据；
[0 0 0]]
"""

""""
[[ 4.  6.  8. 10.]
 [20. 22. 24. 26.]
 [ 0.  1.  2.  3.]]
 
 
 [[ 8. 10. 12. 14.]
 [ 8. 10. 12. 14.]
 [ 4.  5.  6.  7.]]

"""
embed = tf.nn.embedding_lookup_sparse(embedding, idx, None)
embed2 = tf.nn.embedding_lookup(embedding, [1, 2, 3, 4, 5])
mat1 = tf.constant([[0, 1, 1, 1, 1, 1]], dtype=tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.sparse_tensor_to_dense(idx)))

print(sess.run(embed))

mat2 = sess.run(embed2)  ## mat2 是 4*5 矩阵
print("========================================")

print(mat2)
print("========================================")
print(sess.run(tf.reduce_sum(mat2, axis=0)))
print("========================================")
print(sess.run(tf.reduce_sum(mat2, axis=1)))
print(sess.run(tf.matmul(mat1, embedding)))
sess.run(tf.reduce_sum(tf.nn.embedding_lookup(embedding,idx),axis=1))

