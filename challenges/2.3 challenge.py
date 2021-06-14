import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#=== Challenge ============================================================================================

class Perceptron():
    """
    Perceptron model for predicting binary target with canonical learning algorithm
    """

    def __init__(self, w = None, y_classes = None):
        self.w = w
        self.y_classes = y_classes

    def fit(self, X, y, MAXITERS = 100_000):
        """
        Learn separating hyperplane (1-D array of weights, w) via canonical learning algorithm

        :param X: 2-D array with >= 1 column of real-valued features
        :param y: 1-D array of labels; should have two distinct classes
        :param MAXITERS: how many iterations before we give up
        :return: None; set self.y_classes and self.w if a separating hyperplane is found
        """

        ### YOUR CODE HERE ###
        pass

    def predict(self, X):
        """
        Predict on X using this object's w.
        If w•x > 0 we predict y_classes[1], otherwise we predict y_classes[0]

        :param X: 2-D array with >= 1 column of real-valued features
        :return: 1-D array of predicted class labels
        """

        if self.w is None:
            raise AssertionError(f"Need to fit() before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.w) - 1:
            raise AssertionError(
                f"Perceptron was fit on X with {len(self.w) - 1} columns but this X has {X.shape[1]} columns")

        ### YOUR CODE HERE ###
        pass

#=== Test ============================================================================================

p = Perceptron()

### 1-D Test
df1 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_1d.csv')
X, y = df1.drop(columns='y').to_numpy(), df1.y.to_numpy()
p.fit(X = X, y = y)
plot_dataset(X, y, p.w[:-1], p.w[-1])

### 2-D Test
df2 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_2d.csv')
X, y = df2.drop(columns='y').to_numpy(), df2.y.to_numpy()
p.fit(X = X, y = y)
plot_dataset(X, y, p.w[:-1], p.w[-1])

### 3-D Test
df3 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_3d.csv')
X, y = df3.drop(columns='y').to_numpy(), df3.y.to_numpy()
p.fit(X = X, y = y)
plot_dataset(X, y, p.w[:-1], p.w[-1])

### 99-D Test
df99 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_99d.csv')
X, y = df99.drop(columns='y').to_numpy(), df99.y.to_numpy()
p.fit(X = X, y = y)
plot_dataset(X, y, p.w[:-1], p.w[-1])