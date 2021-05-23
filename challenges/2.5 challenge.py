import numpy as np
import pandas as pd

#=== Challenge ============================================================================================

class Perceptron():
    """
    Perceptron that supports multiclass classification via one-vs-rest
    and unseparable classes via the pocket algorithm
    """

    def __init__(self, W = None, y_classes = None):
        """
        Optionally initialize this perceptron with an array of weights and an array of target classes

        :param W: 2-D array where column k is the weight vector associated with class k
        :param y_classes: 1-D array of y classes
        """

        self.W = W
        self.y_classes = y_classes

    def fit(self, X, y, MAXUPDATES=1000, seed=None, verbose=False):
        """
        Fit perceptron on X, y using one-vs-rest and the pocket algorithm
        y should have 2 or more distinct classes

        :param X: 2-D array with >= 1 column of real-valued features
        :param y: 1-D array of labels; should have two distinct classes
        :param MAXUPDATES: how many weight updates to make before quitting, for each submodel
        :param seed: optional random seed
        :param verbose: print progress?
        :return: None; set self.y_classes and self.W
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Determine/validate y_classes
        y_classes = np.unique(y)
        if len(y_classes) < 2:
            AssertionError("y should have at least 2 distinct classes")

        ### YOUR CODE HERE ###
        pass

    def predict(self, X):
        """
        Predict on X using this object's W.
        Use one-vs-rest method to resolve binary predictions (A vs not A, B vs not B, ...)
        to a multiclass prediction.

        :param X: 2-D array with >= 1 column of real-valued features
        :return: 1-D array of predicted class labels
        """

        if self.W is None:
            raise AssertionError(f"Need to fit() a before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.W) - 1:
            raise AssertionError(f"Perceptron was fit on X with {len(self.W) - 1} columns but this X has {X.shape[1]} columns")

        ### YOUR CODE HERE ###
        pass

#=== Test ============================================================================================

### MNIST
mnist_train = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_train.csv')
mnist_test = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_test.csv')
p = Perceptron()
p.fit(
    X = mnist_train.drop(columns='label').to_numpy(),
    y = mnist_train.label.to_numpy(),
    MAXUPDATES = 100,
    seed = 2021,
    verbose = True
)

# Predict and score on mnist_test
test_preds = p.predict(X = mnist_test.drop(columns='label').to_numpy())
np.mean(test_preds == mnist_test.label.to_numpy())