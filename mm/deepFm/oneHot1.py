
import tensorflow as tf

segmentation_gt = [
    [0,0,1,1,1,0,0,0,0],
    [0,0,0,1,0,0,2,2,0],
    [0,3,0,0,0,0,2,2,0],
    [0,1,3,0,0,0,2,0,0],
    [3,3,0,0,2,0,0,0,0],
    [0,0,0,0,2,2,0,0,0]
]
seg_onehot = tf.one_hot(indices=segmentation_gt,depth=4,axis=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(seg_onehot)
    # 查看第三个通道的one-hot编码，我们发现在所有为2的位置，值变为1，其它地方的编码值为0
    print(a[3])
    print(a.shape)
