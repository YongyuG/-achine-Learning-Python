import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.linear_model import LogisticRegression

train_in = pd.read_csv('class_train_in.csv')
train_in = train_in.values[0:1962,1:]
train_out = pd.read_csv('class_train_out.csv')
train_out = train_out.values[0:1962,1:]
test_in = pd.read_csv('class_test_in.csv')
test_in = test_in.values[0:1963,1:]



X=train_in
y = train_out.ravel()

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)


classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.001
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X, y)


nn=classifier.predict(test_in)
k=np.arange(1,1964)
df=pd.DataFrame({'Point_ID': k, 'Output' :nn},index=None, columns=['Point_ID','Output'])

df.to_csv('test_out_dnn.csv',index=False)