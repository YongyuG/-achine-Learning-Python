print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

train_in = pd.read_csv('class_train_in.csv', header=0, delimiter=",")
train_out = pd.read_csv('class_train_out.csv', header=0, delimiter=",")
test_in = pd.read_csv('class_test_in.csv', header=0, delimiter=",")

df=train_in.drop('Point_ID',axis=1)
df1=train_out.drop('Point_ID',axis=1)
df2=test_in.drop('Point_ID',axis=1)train_in = pd.read_csv('class_train_in.csv', header=0, delimiter=",")

train_in_now=pd.Series(df.to_csv('train_in_data.csv', sep=',' , header=False , index=False))
train_out_now=df1.to_csv('train_out_data.csv', sep=',' , header=False,index=False)
test_in_now=df2.to_csv('test_in_data.csv', sep=',' , header=False , index=False)
print(train_in_now)
train_a=pd.read_csv('train_in_data.csv',sep=',',header=None)
train_b=pd.read_csv('train_out_data.csv',sep=',',header=None)
test_in_a=pd.read_csv('test_in_data.csv',sep=',',header=None)