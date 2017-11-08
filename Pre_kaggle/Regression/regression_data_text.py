import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
train_in = pd.read_csv('reg_train_in.csv', sep=",",skiprows=1)
train_out = pd.read_csv('reg_train_out.csv', header=0, sep=",",skiprows=1)
test_in = pd.read_csv('reg_test_in.csv', header=0, sep=",",skiprows=1)
train_in.drop(train_in.columns[0],axis=1,inplace=True)
print(train_in)
def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels