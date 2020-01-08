import numpy as np
from matplotlib import pyplot as plt

predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data = zip(actual, predictions)

sorted_data = sorted(data, key=lambda d: d[1])
sorted_actual = [d[0] for d in sorted_data]
print('Sorted Actual Values', sorted_actual)

cumulative_actual = np.cumsum(sorted_actual)
cumulative_index = np.arange(1, len(cumulative_actual) + 1)

plt.plot(cumulative_index, cumulative_actual)
plt.xlabel('Cumulative Number of Predictions')
plt.ylabel('Cumulative Actual Values')
plt.show()

##基尼系数是这个曲线和直线对角线之间的面积，基尼系数越小，代表样本越平均，基尼系数越大，代表样本的分布越极端化；

"""
actual 代表真实的分值，pred代表预测的分值，如果
基尼系数: 按照实际值进行排序, 然后进行加和累计,这样,
反三角的面积,然后除以
实际值的加和原先是降序排列的，对这个序列最大的累计加和按照这个顺序是最大的，如果用预测值对这个值进行重排序
那么累计加和就必然减少，如果这个减少的值越小，预测的结果就越好；
"""

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]  ##　用pred 值对实际值进行排序，这样对实际值减少了多少？
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses  ##　这里对数据进行标准化；

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


"""
这里是按照预测的分值对真实样本进行排序，如果按照预测出来进行的排序和真实的排序结果很接近，就说明这个排序的结果很好，这里没有采用简单的误差来计算模型的好坏
因为在推荐系统中的rank 模型，我们不关心预测的分数和真实值之间的区别，我们只需要关心预测出来的顺序和真实的顺序是否最大相似；
Gini系数越大，分类效果越好。
"""


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


print(gini(actual, actual))
print(gini(actual, predictions))
print(gini_normalized(actual, predictions))
