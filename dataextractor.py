import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
rawdata = pd.read_csv(link, header=None, encoding='utf-8')

#The perceptron can only handle 2 classe labels. Here we focus only on the first 2 species, setosa and versicolor
#50 measurements for each species are contained in the 100 first data points
y = rawdata.iloc[0:100, 4].values #iloc selects all the dataof column 4, row 0 to 100. values return Series as ndarray or ndarray-like depending on the dtype
y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length
X = rawdata.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

#actually fitting the data

from perceptron import Perceptron

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show() 