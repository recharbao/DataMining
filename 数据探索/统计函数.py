#-*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


catering_sale = '/media/rechar/新加卷/github仓库/DataMining/数据探索/chapter3/demo/data/catering_sale_all.xls'
data = pd.read_excel(catering_sale, index_col = u'日期')

#计算相关系数矩阵
result = data.corr()

print(result)

D = pd.DataFrame([range(1,8), range(2,9)])
result1 = D.corr(method='pearson')
print(result1)

A = pd.DataFrame(np.random.randn(6,5))
print(A)
result2 = A.cov()
print(result2)

#三阶距
print(A.skew())
print(A.describe())

#统计特征函数
M = pd.Series(range(0,20))
result3 = M.cumsum()
print(result3)


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
#plt.figure(figsize=(7,5))
#M.plot(kind='box')
#plt.show()

#x = np.linspace(0, 2*np.pi, 50)
#y = np.sin(x)
#plt.plot(x, y, 'bp--')
#plt.show()

#pie
'''
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True)
plt.axis('equal')
plt.show()
'''

'''
x = np.random.randn(1000)
plt.hist(x, 10)
plt.show()
'''

'''
x = np.random.randn(1000)
F = pd.DataFrame([x, x + 1]).T
F.plot(kind = 'box')
plt.show()

'''

'''
x = pd.Series(np.exp(np.arange(20)))
x.plot(label=u'原始数据图', legend=True)
plt.show()
x.plot(logy=True, label=u'对数数据图', legend=True)
plt.show()
'''

#误差
error = np.random.randn(10)
y = pd.Series(np.sin(np.arange(10)))
y.plot(yerr=error)
plt.show()












