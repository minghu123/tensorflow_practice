import numpy as np
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
b = np.array([[3], [8], [1], [1], [11]])
ohe.fit(b)  # 样本数据
X = np.array([[0], [0], [1], [11]])
c = ohe.transform(X).toarray()
print("c=", c)
