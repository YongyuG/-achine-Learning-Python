print(__doc__)
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import csv
from matplotlib.mlab import rec2csv
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcess
train_in = pd.read_csv('reg_train_in.csv', sep=",")
train_out = pd.read_csv('reg_train_out.csv', sep=",")
test_in = pd.read_csv('reg_test_in.csv',sep=",")

train_in.drop(train_in.columns[0],axis=1,inplace=True)
train_out.drop(train_out.columns[0],axis=1,inplace=True)
test_in.drop(test_in.columns[0],axis=1,inplace=True)
df1=test_in.reindex(index=range(0,2250),columns=list(test_in.columns))

d=df1.fillna(test_in.mean())
print(d)
gp = GaussianProcess(regr='linear',corr='absolute_exponential')
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#regr_1 = DecisionTreeRegressor(max_depth=4)

#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          #n_estimators=300)

X=train_in
y = np.ravel(train_out)
#regr_1.fit(X, y)
#regr_2.fit(X, y)
gp.fit(X, y)

#a = regr_1.predict(d)
#b = regr_2.predict(d)
c=gp.predict(d)
#y_rbf = svr_rbf.fit(X, y).predict(d)
k=np.arange(1,2251)
print(len(k))
#print(len(b))
#   print(len(c))
df=pd.DataFrame({'Point_ID': k, 'Output' :c},index=None, columns=['Point_ID','Output'])

df.to_csv('test_out_rbf.csv',index=False)


