import pandas as pd
data = pd.read_csv('reg_test_in.csv')
train_data = data.values[0:2250,1:]
print(train_data)