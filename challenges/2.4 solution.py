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

class Perceptron():
    """
    Perceptron model for predicting binary target with "pocket" learning algorithm
    """

    def __init__(self, w=None, y_classes=None):
        self.w = w
        self.y_classes = y_classes

    def fit(self, X, y, MAXUPDATES=1000, seed=None, verbose=False):
        """
        Fit perceptron on X, y using the pocket algorithm (variation)

        :param X: 2-D array with >= 1 column of real-valued features
        :param y: 1-D array of labels; should have two distinct classes
        :param MAXUPDATES: how many weight updates to make before quitting
        :param seed: optional random seed
        :param verbose: print progress?
        :return: None; set self.y_classes and self.w
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Determine/validate y_classes
        y_classes = np.unique(y)
        if len(y_classes) != 2:
            AssertionError(f"y should have 2 distinct classes, but instead it has {len(y_classes)}")

        # Convert y to 1-d array of {0, 1} where 0 represents y_classes[0] and 1 represents y_classes[1]
        y01 = (y == y_classes[1]).astype('int64')

        # Initialization
        gen = np.random.default_rng(seed)
        X1 = np.insert(X, X.shape[1], 1, axis=1)
        w_current = np.repeat(0, X1.shape[1])
        w_pocket = np.repeat(0, X1.shape[1])
        accuracy_current = 0
        accuracy_pocket = 0

        # Iterate
        for i in range(MAXUPDATES):

            # Determine yhat for every sample, based on w_current and measure accuracy
            yhat = (np.sign(X1.dot(w_current)) + 1)/2
            accuracy_current = np.mean(yhat == y01)

            # If the accuracy_current is 1, set w_pocket = w_current and break out of this loop
            # If the accuracy_current is better than accuracy_pocket, update w_pocket
            if accuracy_current == 1:
                w_pocket = w_current
                accuracy_pocket = accuracy_current
                if verbose:
                    print(f'Found a separating hyperplane at iteration: {i}')
                break
            elif accuracy_current > accuracy_pocket:
                w_pocket = w_current
                accuracy_pocket = accuracy_current
                if verbose:
                    print(f'Improved hyperplane at iteration: {i}, acurracy: {accuracy_pocket}')

            # Identify a random misclassified training sample and use it to update w_current
            missclassified_idxs = np.nonzero(yhat != y01)[0]
            p = gen.choice(missclassified_idxs, size=1)[0]
            w_current = w_current + (X1[p] if y01[p] == 1 else -X1[p])

        # Update class properties
        self.w = w_pocket
        self.y_classes = y_classes

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
            raise AssertionError(f"Perceptron was fit on X with {len(self.w) - 1} columns but this X has {X.shape[1]} columns")

        X1 = np.insert(X, X.shape[1], 1, axis=1)
        yhat = (X1.dot(self.w) > 0).astype('int64')
        preds = self.y_classes[yhat]

        return preds

#=== Test ============================================================================================

p = Perceptron()

### 1-D Test
df1 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/unseparable_data_1d.csv')
X, y = df1.drop(columns='y').to_numpy(), df1.y.to_numpy()
p.fit(X = X, y = y, verbose=True)
plot_dataset(X, y, p.w[:-1], p.w[-1])

### 2-D Test
df2 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/unseparable_data_2d.csv')
X, y = df2.drop(columns='y').to_numpy(), df2.y.to_numpy()
p.fit(X = X, y = y, verbose=True)
plot_dataset(X, y, p.w[:-1], p.w[-1])

### 3-D Test
df3 = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/unseparable_data_3d.csv')
X, y = df3.drop(columns='y').to_numpy(), df3.y.to_numpy()
p.fit(X = X, y = y, verbose=True)
plot_dataset(X, y, p.w[:-1], p.w[-1])