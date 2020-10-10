#-*- coding: utf-8 -*-
#菜品盈利数据 帕累托图


from __future__ import print_function
import pandas as pd


import sys
reload(sys)
sys.setdefaultencoding("utf-8")


#初始化参数
dish_profit = '/media/rechar/新加卷/github仓库/data/chapter3/demo/data/catering_dish_profit.xls'
data = pd.read_excel(dish_profit, index_col = u'菜品名')
data = data[u'盈利'].copy()
data.sort_values(ascending = False)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

plt.figure()
data.plot(kind = 'bar')
plt.ylabel(u'盈利（元）')
p = 1.0*data.cumsum()/data.sum()
p.plot(color = 'r', secondary_y = True, style = '-o',linewidth = 2)
plt.annotate(format(p[6], '.4%'), xy = (6, p[6]), xytext=(6*0.9, p[6]*0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) 
plt.ylabel(u'盈利（比例）')
plt.show()

