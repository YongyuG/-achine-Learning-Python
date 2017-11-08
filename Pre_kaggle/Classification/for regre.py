import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import l1_min_c
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn import svm

from sklearn.linear_model import LogisticRegression
train_in = pd.read_csv('reg_train_in.csv')
train_in = train_in.values[0:33750,1:]
train_out = pd.read_csv('reg_train_out.csv')
train_out = train_out.values[0:33750,1:]
test_in = pd.read_csv('reg_test_in.csv')
test_in = test_in.values[0:2250,1:]
y_in = pd.read_csv('reg_test_in.csv')
test = y_in.values[0:2250, 1:]

from sklearn.preprocessing import Imputer
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp1.fit(test[0:450])
a=imp1.transform(test[450:750])

imp2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2.fit(test[750:2000])
imp2.transform(test[750:2000])

imp3 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp3.fit(test[1500:2250])
imp3.transform(test[1500:2250])
print(a)

