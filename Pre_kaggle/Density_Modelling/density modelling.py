print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

train_in = pd.read_csv('data.csv', sep=",",header=0)

train_in.drop(train_in.columns[0],axis=1,inplace=True)
#print(train_in)
df1=train_in.reindex(index=range(0,1000),columns=list(['x_ 1','x_ 2','x_ 3','x_ 4','x_ 5','x_ 6','x_ 7','x_ 8','x_ 9','x_11','x_12']))
df2=train_in.reindex(index=range(0,5000),columns=list(['x_ 1','x_ 2','x_ 3','x_ 4','x_ 5','x_ 6','x_ 7','x_ 8','x_ 9','x_11','x_12']))

df5=train_in.reindex(index=range(0,1000),columns=list(['x_ 10','x_ 13','x_ 14']))
df6=train_in.reindex(index=range(0,5000),columns=list(['x_ 10','x_ 13','x_ 14']))


df3=train_in.reindex(index=range(0,2500),columns=list(train_in.columns))
df4=train_in.reindex(index=range(0,5000),columns=list(train_in.columns))

#print(len(df1))
#print(len(df2))
kde = KernelDensity(kernel='gaussian',bandwidth=0.75).fit(df3)
log_dens = kde.score_samples(df4)


print(log_dens)
k=np.arange(1,5001)
df=pd.DataFrame({'Point_ID': k, 'Output' :log_dens},index=None, columns=['Point_ID','Output'])

df.to_csv('test_out1.csv',index=False)
