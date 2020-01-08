import lightgbm as lgb

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

print('Load data...')
df_train = pd.read_csv('D://data/rank/推荐相关数据集/gbdt+lr/train.csv')
df_test = pd.read_csv('D://data/rank/推荐相关数据集/gbdt+lr/test.csv')

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

columns = [temp for temp in df_train.columns if temp != 'target'];

categorical_cols = [temp for temp in df_train.columns if temp not in NUMERIC_COLS and temp != 'target']
print(df_test.head(10))

y_train = df_train['target']  # training label
y_test = df_test['target']  # testing label
X_train = df_train[columns]  # training dataset
X_test = df_test[columns]  # testing dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)  ## 这里对离散性变量怎么处理？？
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

num_leaf = 300
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': num_leaf,
    'num_trees': 20,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                categorical_feature=categorical_cols,
                valid_sets=lgb_train)
lgb.create_tree_digraph()
print('Save model...')
# save model to file
gbm.save_model('model.txt')


print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(X_test, pred_leaf=True)  ##　这个　pred 是预测的叶子节点的位置;
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1


lm = LogisticRegression(penalty='l2', C=0.05)  # logestic model construction
lm.fit(transformed_training_matrix, y_train)  # fitting the data
y_pred_test = lm.predict_proba(transformed_testing_matrix)  # Give the probabilty on each label

print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(
    ((1 + y_test) / 2 * np.log(y_pred_test[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_test[:, 1])))
print("Normalized Cross Entropy " + str(NE))
