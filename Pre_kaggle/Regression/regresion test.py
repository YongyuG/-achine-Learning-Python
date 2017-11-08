import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostRegressor

train_in = pd.read_csv('reg_train_in.csv', sep=",")
train_out = pd.read_csv('reg_train_out.csv', sep=",")
test_in = pd.read_csv('reg_test_in.csv',sep=",")

test_in.fillna(test_in.mean())

train_in.drop(train_in.columns[0],axis=1,inplace=True)
train_out.drop(train_out.columns[0],axis=1,inplace=True)
test_in.drop(test_in.columns[0],axis=1,inplace=True)

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

X=train_in
y=train_out

df1=test_in.reindex(index=range(1,2250),columns=list(test_in.columns))
df1.dropna(how='any')
print(df1)
a=regr_1.fit(X, y)
b=regr_2.fit(X, y)


k=range(1,2250)

y_1 = regr_1.predict(test_in)
y_2 = regr_2.predict(test_in)

df=pd.DataFrame({'Point_ID': k, 'Output' :a},index=None, columns=['Point_ID','Output'])
print(df)
df.to_csv('test_out3.csv',index=False)
