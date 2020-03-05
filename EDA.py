import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_processed/combined.csv',nrows=5000)
# print(df.head(3))

print(df)

correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
plt.show()



df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df)
plt.show()

df.hist()
plt.show()
