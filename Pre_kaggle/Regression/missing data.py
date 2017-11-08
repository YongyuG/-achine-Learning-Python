import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn.gaussian_process import GaussianProcess
from matplotlib.mlab import rec2csv
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

data = pd.read_csv('reg_test_in.csv')
test_in = data.values[0:2250,1:]

rng = np.random.RandomState(0)



X_full=test_in
n_samples = 2250
n_features = 14
#print(n_samples,n_features)
#print(X_full.shape)

# Estimate the score on the entire dataset, with no missing values
#estimator = RandomForestRegressor(random_state=0, n_estimators=100)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)


# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
#y_missing = y_full.copy()
#estimator = Pipeline([("imputer", Imputer(missing_values=0,
                      #                     strategy="mean",
                      #                     axis=0)),
                      # ("forest", RandomForestRegressor(random_state=0,
                      #                                  n_estimators=100))])
print(X_missing)
train_in = pd.read_csv('reg_train_in.csv')
train_in = train_in.values[0:33750,1:]
train_out = pd.read_csv('reg_train_out.csv')
train_out = train_out.values[0:33750,1:]

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_in)
X_test=imp.transform(X_full)



svr_rbf = SVR(kernel='poly', C=1e3, degree=3)


X=train_in

y = train_out.ravel()
print(train_in.shape,train_out.shape)
svr_rbf.fit(X, y)
y_rbf=svr_rbf.predict(X_test)
k=np.arange(1,2251)
df=pd.DataFrame({'Point_ID': k, 'Output' :y_rbf},index=None, columns=['Point_ID','Output'])

df.to_csv('test_out_rbf_11111.csv',index=False)

