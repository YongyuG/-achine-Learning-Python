from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity





train_in = pd.read_csv('data.csv', sep=",",header=0)

train_in.drop(train_in.columns[0],axis=1,inplace=True)

n_digits = len(np.unique(train_in))

df3=train_in.reindex(index=range(0,5000),columns=list(['x_ 1']))
df4=train_in.reindex(index=range(0,5000),columns=list(train_in.columns))




kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df3)
print(kde.shape)
plt.figure(1)

plt.clf()

plt.plot(df3,kde)
plt.xticks(())
plt.yticks(())
plt.show()
