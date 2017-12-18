from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
import seaborn as sns
train_in = pd.read_csv('data.csv', sep=",",header=0)

train_in.drop(train_in.columns[0],axis=1,inplace=True)

n_digits = len(np.unique(train_in))

df2=train_in.reindex(index=range(0,5000),columns=list(['x_ 1','x_ 2']))
df3=train_in.reindex(index=range(0,3000),columns=list(train_in.columns))
df4=train_in.reindex(index=range(0,5000),columns=list(train_in.columns))
train_in = pd.read_csv('data.csv')
train_in = train_in.values[0:5000,1:]
i=0

reduced_data = PCA(n_components=1).fit_transform(df4)
sns.distplot(reduced_data, kde=False, rug=True)



plt.show()