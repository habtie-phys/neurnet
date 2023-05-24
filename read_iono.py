import os
import numpy as np
import pandas as pd
#
wd = os.getcwd()
file = r'data/ionosphere.data'
fp = os.path.join(wd, file)
ds = pd.read_csv(fp)
#
ds.columns = range(35)
s = ds[34].to_numpy()
data = ds[range(34)].to_numpy()
s[s=='b'] = 0
s[s=='g'] = 1
#
inds = np.arange(s.size).astype(int)
train_idx, test_idx = inds[:150], inds[150:]
x_train, x_test = data[train_idx,:], data[test_idx,:]
y_train, y_test = s[train_idx], s[test_idx]
#
class iono_data:
    def get_train(self):
        x = x_train.T
        y = y_train[np.newaxis,:].astype(float)
        return (x, y)
    def get_test(self):
        x = x_test.T
        y = y_test[np.newaxis,:].astype(float)
        return (x, y)
