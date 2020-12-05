#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
import numpy as np

e = pd.Series([1, 1, 1, 1, 2, 2, 3])
print(e)
res = e.unique()
print(res)
res1 = np.unique(e)
print(res1)