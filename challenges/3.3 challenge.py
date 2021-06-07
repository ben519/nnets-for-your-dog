import numpy as np
import pandas as pd

#=== Challenge ============================================================================================

def logistic(x):
    """
    standard logistic function

    uses the identity: 1/(1 + e^(-x)) = e^x/(e^x + 1)
    to prevent double precision issues when x is a big negative number

    :param x: numpy array
    :return: 1/(1 + e^(-x))
    """

    mask = x > 0
    y = np.full(shape=x.shape, fill_value=np.nan)
    y[mask] = 1 / (1 + np.exp(-(x[mask])))
    y[~mask] = np.exp(x[~mask]) / (np.exp(x[~mask]) + 1)
    return y


def logloss(yhat, y):
    """
    Calculate log loss vector

    :param yhat: numpy array of predicted probabilities in range [0,1]
    :param y: numpy array of true probabilities in range [0, 1]
    :return: numpy array of log loss for every (yhat_i, y_i)
    """

    return -(y * np.log(yhat) + (1-y) * np.log(1-yhat))


class NNet():
    """
    NNet with gradient descent
    """

    def __init__(self, W1=None, W2=None, y_classes=None):
        """
        Initialization

        :param W1: optional weight matrix, W1 (2-D numpy array)
        :param W2: optional weight matrix, W2 (2-D numpy array)
        :param y_classes: optional array of y_classes (1-D numpy array with 2 elements)
        """

        self.W1 = W1
        self.W2 = W2
        self.y_classes = y_classes

    def fit(self, X, y, hiddenNodes, stepSize=0.01, ITERS=100, seed=None):
        """
        Find the best weights via gradient descent

        :param X: training features
        :param y: training labels. Should have exactly 2 classes
        :param hiddenNodes: How many hidden layer nodes to use, excluding bias node
        :param stepSize: AKA "learning rate" AKA "alpha" used in gradient descent
        :param ITERS: How many gradient descent steps to make?
        :return: None. Update self.y_classes, self.W1, self.W2
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Determine/validate y_classes
        y_classes = np.unique(y)
        if len(y_classes) != 2:
            AssertionError(f"y should have 2 distinct classes, but instead it has {len(y_classes)}")

        pass

    def predict(self, X, type='probs'):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :return: if type = 'probs' then probabilities else if type = 'classes' then classes
        """

        if self.W1 is None:
            raise AssertionError(f"Need to fit() a before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.W1) - 1:
            raise AssertionError(f"Perceptron was fit on X with {len(self.W1) - 1} columns but this X has {X.shape[1]} columns")

        X1 = np.insert(X/255, obj=X.shape[1], values=1, axis=1)
        Z1 = X1 @ self.W1
        X2 = np.insert(logistic(Z1), obj=Z1.shape[1], values=1, axis=1)
        Z2 = X2 @ self.W2
        yhat_probs = logistic(Z2)[:, 0]

        if type == 'probs':
            return yhat_probs
        elif type == 'classes':
            yhat_classes = self.y_classes[(yhat_probs > 0.5).astype('int64')]
            return yhat_classes

#=== Test ============================================================================================

# Load simple images data
train = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_test.csv")

# Initialize & fit neural network
nn = NNet()
nn.fit(
    X = train.drop(columns='label').to_numpy(),
    y = (train.label == 'checkerboard').to_numpy(),
    hiddenNodes = 4,
    stepSize = 0.3,
    ITERS = 10_000,
    seed = 123
)

# Evaluate on test data
preds = nn.predict(X = test.drop(columns='label').to_numpy())
(preds == (test.label == 'checkerboard')).mean()
