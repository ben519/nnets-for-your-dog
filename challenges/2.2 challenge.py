import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#=== Helpers ============================================================================================

def plot_dataset(X, y, w=None, b=None):
    """
    Plot X, y data

    :param X: 2-D array of features (with 1, 2, or 3 columns)
    :param y: 1-D array of lables
    :return: Axes object
    """

    colors = ListedColormap(['r', 'b', 'g'])

    if X.shape[1] == 1:
        scatter = plt.scatter(X[:, 0], np.repeat(0, X.size), c=y, cmap=colors)

        if w is not None and b is not None:
            x1 = -b/w[-1]
            plt.scatter(x1, 0, marker = 'x')

    elif X.shape[1] == 2:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors)

        if w is not None and b is not None:
            x1 = np.array([0, 1])
            x2 = -(w[0] * x1 + b)/w[-1]
            plt.axline(xy1=(x1[0], x2[0]), xy2=(x1[1], x2[1]))

    elif X.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=colors)

        if w is not None and b is not None:
            x1 = X[:, 0]
            x2 = X[:, 1]
            x3 = -(X[:, [0, 1]].dot(w[[0, 1]]) + b)/w[-1]
            ax.plot_trisurf(x1, x2, x3)

    else:
        raise AssertionError("Can't plot data with >3 dimensions")

    # insert legend
    plt.legend(*scatter.legend_elements())

    return plt.gca()

#=== Challenge ============================================================================================

def guess_hyperplane(X, y, MAXGUESSES=100_000):
    """
    Given a dataset of features X and binary labels y which we assume to be linearly separable,
    guess random hyperplanes until we get one that separates the data.

    :param X: 2-D array with >= 1 column of real-valued features
    :param y: 1-D array of labels in {0, 1}
    :param MAXGUESSES: how many times to guess before we give up
    :return: tuple of (w, b) where w is a 1-D array of weights and b is an offset
    """

    ### YOUR CODE HERE ###
    pass


#=== Test ============================================================================================

### 1-D Test
df1 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_1d.csv')
X, y = df1.drop(columns='y').to_numpy(), df1.y.to_numpy()
w, b = guess_hyperplane(X, y)
plot_dataset(X, y, w, b)

### 2-D Test
df2 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_2d.csv')
X, y = df2.drop(columns='y').to_numpy(), df2.y.to_numpy()
w, b = guess_hyperplane(X, y)
plot_dataset(X, y, w, b)

### 3-D Test
df3 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_3d.csv')
X, y = df3.drop(columns='y').to_numpy(), df3.y.to_numpy()
w, b = guess_hyperplane(X, y)
plot_dataset(X, y, w, b)

### 99-D Test
df99 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/separable_data_99d.csv')
X, y = df99.drop(columns='y').to_numpy(), df99.y.to_numpy()
w, b = guess_hyperplane(X, y)
plot_dataset(X, y, w, b)