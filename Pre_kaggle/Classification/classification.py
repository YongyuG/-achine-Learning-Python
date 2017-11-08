print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import csv
from matplotlib.mlab import rec2csv
from sklearn.ensemble import RandomForestClassifier


train_in = pd.read_csv('class_train_in.csv', sep=",")
train_out = pd.read_csv('class_train_out.csv', sep=",")
test_in = pd.read_csv('class_test_in.csv',sep=",")

train_in.drop(train_in.columns[0],axis=1,inplace=True)
train_out.drop(train_out.columns[0],axis=1,inplace=True)
test_in.drop(test_in.columns[0],axis=1,inplace=True)
# Generate train data
# Our dataset and targets


# fit the model
X = train_in

y = np.ravel(train_out)
#clf = QuadraticDiscriminantAnalysis()
clf = svm.SVC(kernel='poly',degree=3.8,gamma=0.8, probability=True,random_state=0)
#neigh = KNeighborsClassifier(n_neighbors=5)
#clf1=GaussianNB()
clf.fit(X,y)


#neigh.fit(X, y)
#clf1.fit(X,y)
#a=clf.predict(test_in)
#a1=clf1.predict(test_in)
#a1=neigh.predict(test_in)
#o=np.transpose(np.matrix(a))
probs=clf.predict_proba(test_in)
output=[]
for i in probs:
    print(i[0])
    output.append(i[0])
print(len(output))
np.array(output).reshape(1963,1)
u=np.transpose(np.matrix(range(1,1963)))

k=np.arange(1,1964)
m=range(1,1963)


#df=pd.DataFrame({'Point_ID': k, 'Output' :a},index=None, columns=['Point_ID','Output'])
df=pd.DataFrame({'Point_ID': k, 'Output' :output},index=None, columns=['Point_ID','Output'])
print(df)
df.to_csv('test_out_pro.csv',index=False)


#np.savetxt("class_test_out.csv", [[k,a]],fmt='%f %f',header='Point_ID,Output', delimiter=",",comments='')
#np.savetxt("class_test_out.csv", data.reshape(1963,2),fmt='%i ,%i',header='Point_ID,Output', delimiter=",",comments='')

