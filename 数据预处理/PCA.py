#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
from scipy.interpolate import lagrange

inputfile = '/media/rechar/新加卷/github仓库/data/chapter4/demo/data/principal_component.xls'
outputfile = './result.xls'

data = pd.read_excel(inputfile, header = None)


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
print(pca.components_)
print(pca.explained_variance_ratio_)

pca = PCA(3)
pca.fit(data)
res = pca.transform(data)
print(res)


