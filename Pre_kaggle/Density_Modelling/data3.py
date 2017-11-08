from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
from sklearn import datasets


train_in = pd.read_csv('data.csv')
train_in = train_in.values[0:5000,0:]
x=train_in[0:5000,10]
iris = datasets.load_iris()
print(iris.target)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()

sns.set(color_codes=True)
sns.distplot(x, kde=False, rug=True);

plt.show()