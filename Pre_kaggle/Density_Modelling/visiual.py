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

reduced_data = PCA(n_components=2).fit_transform(df4)

#reduced_data=train_in[0:5000,0:2]
print(train_in[0:5000,2:4])
#kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df3)

kmeans = KMeans(init='k-means++', n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(reduced_data)
#Z=kde.score(np.c_[xx.ravel(), yy.ravel()])
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)


Z = Z.reshape(xx.shape)
plt.figure(1)

plt.clf()

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

