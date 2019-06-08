import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('ClientDetails.csv', header=0)
data = data.dropna()

data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
data2 = pd.get_dummies(data, columns=['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)


f, ax = plt.subplots(figsize=(11, 9))                   # Set up the matplotlib figure
cmap = sns.diverging_palette(220, 10, as_cmap=True)     # Generate a custom colormap
sns.heatmap(data2.corr(), cmap=cmap)

plt.show()

# https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/

import statsmodels.api as sm
train_cols = data2.columns[1:]
print(train_cols)
logit = sm.Logit(data2['y'], data2[train_cols])
# fit the model
result = logit.fit()
print(result.summary2())