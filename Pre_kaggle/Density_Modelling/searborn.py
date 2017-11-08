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

#f,ax=plt.subplots(7,2,sharex=True,sharey=True)
fig=plt.figure()
ax1=fig.add_subplot(721)
ax1=sns.distplot(x1, hist=False, rug=True)
ax2=fig.add_subplot(722)
ax2=sns.distplot(x2, hist=False, rug=True)
ax3=fig.add_subplot(723)
ax3=sns.distplot(x3, hist=False, rug=True)
ax4=fig.add_subplot(724)
ax4=sns.distplot(x4, hist=False, rug=True)
ax5=fig.add_subplot(725)
ax5=sns.distplot(x5, hist=False, rug=True)
ax6=fig.add_subplot(726)
ax6=sns.distplot(x6, hist=False, rug=True)
ax7=fig.add_subplot(727)
ax7=sns.distplot(x7, hist=False, rug=True)
ax8=fig.add_subplot(728)
ax8=sns.distplot(x8, hist=False, rug=True)
ax9=fig.add_subplot(7,2,9)
ax9=sns.distplot(x9, hist=False, rug=True)
ax10=fig.add_subplot(7,2,10)
ax10=sns.distplot(x10, hist=False, rug=True)
ax11=fig.add_subplot(7,2,11)
ax11=sns.distplot(x11, hist=False, rug=True)
ax12=fig.add_subplot(7,2,12)
ax12=sns.distplot(x12, hist=False, rug=True)
ax13=fig.add_subplot(7,2,13)
ax13=sns.distplot(x13, hist=False, rug=True)
ax14=fig.add_subplot(7,2,14)
ax14=sns.distplot(x14, hist=False, rug=True)



plt.show()