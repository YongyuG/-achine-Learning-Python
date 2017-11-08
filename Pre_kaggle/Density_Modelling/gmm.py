import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
print(__doc__)
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


    train_in = pd.read_csv('data.csv', sep=",",header=0)

    train_in.drop(train_in.columns[0],axis=1,inplace=True)


    df3=train_in.reindex(index=range(0,3000),columns=list(train_in.columns))
    df4=train_in.reindex(index=range(0,5000),columns=list(train_in.columns))

clf = mixture.GMM(n_components=30, covariance_type='full',n_iter=100)
clf.fit(df3)

log_dens=clf.score(df4)
k=np.arange(1,5001)
df=pd.DataFrame({'Point_ID': k, 'Output' :log_dens},index=None, columns=['Point_ID','Output'])

df.to_csv('test_out_gmm1.csv',index=False)

