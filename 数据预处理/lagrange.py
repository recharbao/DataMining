#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
from scipy.interpolate import lagrange

inputfile = '/media/rechar/新加卷/github仓库/data/chapter4/demo/data/catering_sale.xls'

data = pd.read_excel(inputfile)



data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] =  None
print(data)

def ploy(s, n, k = 5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + k + 1))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)



for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ploy(data[i], j)



print(data)   





