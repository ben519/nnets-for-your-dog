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

    :param X: 2-D array with >= 1 column of features
    :param y: 1-D array of labels in {0, 1}
    :param MAXGUESSES: how many times to guess before we give up
    :return: tuple of (w, b) where w is a 1-D array of weights and b is an offset
    """

    # Set up a random number generator
    gen = np.random.default_rng()

    # In order to guess hyperplanes that have a reasonable chance of separating the data, we
    # 1) Guess random weights in the range [-1000, 1000]
    # 2) Identify a bounding box around the data in X
    # 3) Pick a random point P in the bounding box
    # 4) Calculate b such that the hyper-plane goes passes through it

    # Repeat until we either find a separating hyperplane or we're tired of guessing
    for i in range(MAXGUESSES):
        # Determine X bounds (bounding box)
        X_mins = X.min(axis=0)
        X_maxs = X.max(axis=0)

        # Guess weights
        w = gen.uniform(low=-1000, high=1000, size=X.shape[1])

        # Calculate b such that hyperplane goes through a random point inside X's bounding box
        P = gen.uniform(low=X_mins, high=X_maxs)
        b = -P.dot(w)

        # Check if the hyperplane separates the data
        yhat = np.sign(X.dot(w) + b)
        yhat = (yhat + 1) / 2  # transform (1, 0, -1) -> (1, 0.5, 0)
        if (np.all(yhat == y)):
            break

    # Check outcome based on i
    if i == (MAXGUESSES - 1):
        print("Out of guesses. Maybe this data isn't linearly separable..?")
        return None
    else:
        print(f"Found a separating hyperplane in {i + 1} guesses!")
        return (w, b)

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