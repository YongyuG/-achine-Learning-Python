# Machine-Learning-Python
This repository is going to sotre Machine Learning project coded by Python

Pre_Kaggle:
====
This folder includes machine learning project which I did one years ago.
It comprises three Kaggle competitions in three different settings: classification, regression,
and density modelling.

Classification:
----
Description of training data. The training dataset comprises 1,962 data points with 265
dimensional inputs and binary class labels. The training inputs and outputs are available as
separate comma separated value (CSV) files. The first row of each CSV file contains the column
names and the first column contains the data point index (running from 1 to 1,962).

Description of test data. The test dataset comprises 1,963 data points with 265 dimensional
input features. The binary class labels are missing. The goal is to predict the probability that
each missing class label is a 1. The test inputs are also available as a CSV file in the same
format as the training data.

Submission of predictions. For each test data point you must predict the probability that its
output takes the value 1. These predictions should be submitted in the same format as the
training outputs. The first row must contain column names (Point_ID, Output). Below this
row, the first column must contain the data point index (that runs from 1 through to 1,963)
and the second column must contain the predictions (numbers between 0 and 1).

Description of evaluation metric. The Area Under the Receiver Operator Characteristic (AU-
ROC) will be used for evaluation 

Regression:
----
Description of training data. The training dataset comprises 33,750 data points with 14
dimensional inputs and one dimensional real-valued outputs. The training inputs and outputs
are available as separate CSV files as before.

Description of test data. The test dataset comprises 2,250 data points with 14 dimensional
input features. The outputs are missing and the goal is to predict them. In addition, at some
of the test points a subset of the inputs is also missing. The test inputs are also available as
a CSV file in the same format as the training data. Missing inputs are indicated by the value
NaN.

Results format of predictions. For each test data point you must predict the missing real-valued
output. These predictions should be submitted in the same format as the training outputs.
The first row must contain column names (Point_ID, Output). Below this row, the first
column must contain the data point index (that runs from 1 through to 2,250) and the second
column must contain the predictions (floating point numbers).

Description of evaluation metric. The Root Mean Squared Error (RMSE) will be used for
evaluation. The RMSE is defined as the average error between the predictions (ŷ) and ground truth outputs (y).
Outputs associated with missing inputs contribute equally to those without

Density Modelling:
----
Description of training data. The dataset comprises 5,000 data points each of 14 dimensions
drawn from an unknown probability density. The data are available as a CSV file. The goal is
to estimate the log-probability density at each of these points. Note that three of the variables
are binary and the rest are real valued.

Description of test data. There are no separate test data for this competition as it is an
unsupervised learning task. Predictions should be made at the training data locations.

Results formatof predictions. For each training data point you must predict the log-probability
density. These predictions should be submitted in the same format as the example submission.
The first row must contain column names (Point_ID, Output). Below this row, the first column
must contain the data point index (that runs from 1 through to 5,000) and the second column
must contain the log-density predictions (floating point numbers).

Description of evaluation metric. The correlation (Pearson’s) with the true log-density will
be used for evaluation.
