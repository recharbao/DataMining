#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np

datafile = '/media/rechar/新加卷/github仓库/DataMining/数据预处理/data/normalization_data.xls'

data = pd.read_excel(datafile, header=None)

print((data - data.min())/(data.max() - data.min()))

print((data - data.mean())/data.std())

print((data/10 ** np.ceil(np.log10(data.abs().max()))))

