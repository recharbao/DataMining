#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd

inputFile = 'data/sales_data.xls'
data = pd.read_excel(inputFile, index_col=u'序号')
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
print(data)

x = data.iloc[:, : 3].as_matrix().astype(int)
y = data.iloc[:, 3].as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC

dtc = DTC(criterion='entropy')
dtc.fit(x, y)

from sklearn.tree import export_graphviz
x = pd.DataFrame(x)
print(x)

with open("tree.dot", 'w') as f:
    f = export_graphviz(dtc, feature_names=x.columns, out_file= f)

