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

print(df1.index('nan'))