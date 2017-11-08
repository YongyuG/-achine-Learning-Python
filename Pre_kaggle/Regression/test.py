import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

train_in = pd.read_csv('reg_train_in.csv', header=0, delimiter=",")
train_out = pd.read_csv('reg_train_out.csv', header=0, delimiter=",")
test_in = pd.read_csv('reg_test_in.csv', header=0, delimiter=",")

df=train_in.drop('Point_ID',axis=1)
df1=train_out.drop('Point_ID',axis=1)
df2=test_in.drop('Point_ID',axis=1)

train_in_now=df.to_csv('train_in_data.csv', sep=',' , header=False , index=False)
train_out_now=df1.to_csv('train_out_data.csv', sep=',' , header=False,index=False)
test_in_now=df2.to_csv('test_in_data.csv', sep=',' , header=False , index=False)

train_a=pd.read_csv('train_in_data.csv',sep=',')
train_b=pd.read_csv('train_out_data.csv',sep=',')
test_in_a=pd.read_csv('test_in_data.csv',sep=',')

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
X=train_a
y=train_b
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict

X_test = test_in
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
#