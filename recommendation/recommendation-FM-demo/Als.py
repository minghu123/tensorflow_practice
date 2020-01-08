"""
手动实现ALS最小二乘算法
"""

import pandas as pd
import numpy as np

from scipy.sparse import csr  ## 用来构建矩阵;

# 首先读取数据
cols = ['user', 'item', 'rating', 'timestamp']

train = pd.read_csv('data/ua.base', delimiter="\t", names=cols)  # type:pandas.core.frame.DataFrame
test = pd.read_csv('data/ua.test', delimiter="\t", names=cols)

arr1 = np.zeros((12, 3), dtype=int);

k_factor = 40  ##　设置用户矩阵和物品矩阵的维度；

##：将train 转换为一个用户物品矩阵,矩阵的元素是对其的评分;先确定矩阵的维度

user = train['user'].values  # type:numpy.ndarray

item = train['item'].values

## 这里产生一个用户和下标的对应关系;


userTransFrom = np.zeros(user.shape, dtype=int)

userDic = dict()
userDic[0] = user[0]
i = 0;
j = 0;
for tmp in user:
    index = userDic.get(tmp)
    if index == None:
        i = i + 1
        userTransFrom[j] = i;
        userDic[tmp] = i
    else:
        userTransFrom[j] = index
    j = j + 1;
print(userTransFrom)

itemDic = dict()
itemDic[0] = item[0];
itemTransForm = np.empty(item.shape, dtype=int)
i = 0
j = 0
for tmp in item:
    index = itemDic.get(tmp)
    if index == None:
        i = i + 1
        itemTransForm[j] = i
        itemDic[tmp] = i
    else:
        itemTransForm[j] = tmp
    j = j + 1
print(itemTransForm)

## 构建矩阵;
userItem = csr.csr_matrix((train['rating'], (userTransFrom, itemTransForm)),
                          shape=(userDic.__len__(), itemDic.__len__()))

print(userItem.todense())
##构建出来一个用户物品矩阵了，下面开始使用矩阵分解的方法来构建　 ,

# 先初始化两个随机用户矩阵
lenUser = userDic.__len__();
lenItem = itemDic.__len__();
userF = np.random.random((lenUser, k_factor))
itemF = np.random.random((lenItem, k_factor))


##　为了解决 A *x =B 的问题 ，解决出来向量x;

##这个 lamaba 是个什么东西??
def als_step(latent_vectors,
             fixed_vecs,
             ratings,
             _lambda,
             type='user'):
    """
    One of the two ALS steps. Solve for the latent vectors
    specified by type.
    """
    if type == 'user':
        # Precompute
        YTY = fixed_vecs.T.dot(fixed_vecs)
        lambdaI = np.eye(YTY.shape[0]) * _lambda

        ##　这里是对每个人进行训练，也就是分批训练吧，就是物品矩阵不动的时候，每个用户矩阵之间不相互影响；；

        for u in range(latent_vectors.shape[0]):
            latent_vectors[u, :] = np.linalg.solve((YTY + lambdaI),
                                                   ratings[u, :].dot(fixed_vecs).T).T
    elif type == 'item':
        # Precompute
        XTX = fixed_vecs.T.dot(fixed_vecs)
        lambdaI = np.eye(XTX.shape[0]) * _lambda

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i, :] = np.linalg.solve((XTX + lambdaI),
                                                   ratings[:, i].T.dot(fixed_vecs).T).T
    return latent_vectors


# 设置正则化项；
userReg = np.eye(lenItem, lenItem, dtype=float) * 0.1
itemReg = np.eye(lenUser, lenUser, dtype=float) * 0.1


for i in range(100):
    userF = als_step(userF, itemF, ratings=userItem, _lambda=0.1, type="user");
    itemF = als_step(itemF, userF, ratings=userItem, _lambda=0.1, type="item")
    #计算损失函数；
    predict=userF.dot(itemF.T)
    err=np.array(userItem-predict)
    err=np.sum(err*err)
    ru=np.sum(userF*userF)*0.1
    ri=np.sum(itemF*itemF)*0.1
    print(err+ru+ri)


