#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd

filename = 'data/bankloan.xls'
data = pd.read_excel(filename)


print(data)
x = data.iloc[:, : 8].as_matrix()
print(x)
y = data.iloc[:, 8].as_matrix()
print(y)

#逻辑回归
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

#随机逻辑回归模型
rlr = RLR()
rlr.fit(x, y)
print(rlr.get_support())
print(rlr.scores_)
#有效特征
tmp = rlr.get_support()
dataCol = data.columns[: 8]
print(dataCol)
print(dataCol[tmp == True])

x = data[dataCol[tmp == True]].as_matrix()
print(x)

lr = LR()
lr.fit(x, y)
print(lr.score(x, y))

