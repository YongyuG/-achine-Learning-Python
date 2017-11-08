import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
train_in = pd.read_csv('data.csv')
train_in = train_in.values[0:5000,1:]

result = pd.read_csv('test_out_gmm1.csv')
result = result.values[0:5000,1:]


x1=train_in[0:5000,0:1]
x2=train_in[0:5000,1:2]
x3=train_in[0:5000,2:3]
x4=train_in[0:5000,3:4]
x5=train_in[0:5000,4:5]
x6=train_in[0:5000,5:6]
x7=train_in[0:5000,6:7]
x8=train_in[0:5000,7:8]
x9=train_in[0:5000,8:9]
x10=train_in[0:5000,9:10]
x11=train_in[0:5000,10:11]
x12=train_in[0:5000,11:12]
x13=train_in[0:5000,12:13]
x14=train_in[0:5000,13:14]
sns.set(style="white", palette="muted", color_codes=True)


f, axes = plt.subplots(7, 2, figsize=(20, 20), sharex=True)
sns.distplot(x1, hist=False, rug=True, color="r", ax=axes[0, 0])
sns.distplot(x2, hist=False, rug=True, color="r", ax=axes[0, 1])
sns.distplot(x3, hist=False, rug=True, color="r", ax=axes[1, 0])
sns.distplot(x4, hist=False, rug=True, color="r", ax=axes[1, 1])
sns.distplot(x5, hist=False, rug=True, color="r", ax=axes[2, 0])
plt.setp(axes, yticks=[])
plt.tight_layout()

plt.show()

