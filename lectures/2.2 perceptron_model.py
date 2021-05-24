import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, linewidth=999)

#=== Perceptron class ============================================================================================

class Perceptron():
    """
    Perceptron model for predicting binary target with random guessing fit() procedure
    """

    def __init__(self, w = None, b = None, y_classes = None):
        self.w = w
        self.b = b
        self.y_classes = y_classes

    def fit(self, X, y, MAXGUESSES = 100_000, seed = None):
        """
        Randomly guess hyperplanes until we get one that separates the data,
        or until we exhaust our guesses

        :param X: 2-D array with >= 1 column of real-valued features
        :param y: 1-D array of labels; should have two distinct classes
        :param MAXGUESSES: how many times to guess before we give up
        :param seed: optional random seed
        :return: None; set self.y_clsses, self.w and self.b if a separating hyperplane is found
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Determine/validate y_classes
        y_classes = np.unique(y)
        if len(y_classes) != 2:
            AssertionError(f"y should have 2 distinct classes, but instead it has {len(y_classes)}")

        # Convert y to 1-d array of {0, 1} where 0 represents class y_classes[0] and 1 represents y_classes[1]
        y01 = (y == y_classes[1]).astype('int64')

        # Set up a random number generator
        gen = np.random.default_rng(seed=seed)

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
            yhat = (np.sign(X.dot(w) + b) + 1) / 2
            if (np.all(yhat == y01)):
                break

        # Check outcome based on i
        if i == (MAXGUESSES - 1):
            print("Out of guesses. Maybe this data isn't linearly separable..?")
        else:
            print(f"Found a separating hyperplane in {i + 1} guesses!")
            self.w = w
            self.b = b
            self.y_classes = y_classes


    def predict(self, X):
        """
        Predict on X using this object's w and b.
        If w•x + b > 0 we predict y_classes[1], otherwise we predict y_classes[0]

        :param X: 2-D array with >= 1 column of real-valued features
        :return: 1-D array of predicted class labels
        """

        if self.w is None:
            raise AssertionError(f"Need to fit() a before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.w):
            raise AssertionError(f"Perceptron was fit on X with {len(self.w)} columns but this X has {X.shape[1]} columns")

        yhat = (X.dot(self.w) + self.b > 0).astype('int64')
        preds = self.y_classes[yhat]

        return preds

