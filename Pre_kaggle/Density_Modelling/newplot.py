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
ax1=sns.distplot(x1)
ax1.set(title='Dimension 1 distribution')
ax2=fig.add_subplot(722)
ax2=sns.distplot(x2)
ax2.set(title='Dimension 2 distribution')
ax3=fig.add_subplot(723)
ax3=sns.distplot(x3)
ax3.set(title='Dimension 3 distribution')
ax4=fig.add_subplot(724)
ax4=sns.distplot(x4)
ax4.set(title='Dimension 4 distribution')
ax5=fig.add_subplot(725)
ax5=sns.distplot(x5)
ax5.set(title='Dimension 5 distribution')
ax6=fig.add_subplot(726)
ax6=sns.distplot(x6)
ax6.set(title='Dimension 6 distribution')
ax7=fig.add_subplot(727)
ax7=sns.distplot(x7)
ax7.set(title='Dimension 7 distribution')
ax8=fig.add_subplot(728)
ax8=sns.distplot(x8)
ax8.set(title='Dimension 8 distribution')
ax9=fig.add_subplot(7,2,9)
ax9=sns.distplot(x9)
ax9.set(title='Dimension 9 distribution')
ax10=fig.add_subplot(7,2,10)
ax10=sns.distplot(x10)
ax10.set(title='Dimension 10 distribution')
ax11=fig.add_subplot(7,2,11)
ax11=sns.distplot(x11)
ax11.set(title='Dimension 11 distribution')
ax12=fig.add_subplot(7,2,12)
ax12=sns.distplot(x12)
ax12.set(title='Dimension 12 distribution')
ax13=fig.add_subplot(7,2,13)
ax13=sns.distplot(x13)
ax13.set(title='Dimension 13 distribution')
ax14=fig.add_subplot(7,2,14)
ax14=sns.distplot(x14)
ax14.set(title='Dimension 14 distribution')



plt.show()