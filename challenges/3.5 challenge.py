import numpy as np
import pandas as pd

#=== Challenge ============================================================================================

def softmax(x):
    """
    Calculate row-wise softmax

    :param x: 2d array where (i,j) gives the jth input value for the ith sample
    :return: 2d array with the same shape as the input, with softmax applied to each row-vector
             As a result, the elements in each row can be interpretted as probabilities that sum to one
    """

    return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]


def one_hot(x):
    """
    One-hot-encode an array

    :param x: 1d array where element (i) gives the true label for sample i
    :return: tuple of (onehot, classes) where:
             - onehot is a NxK array where N = len(x), K = len(np.unique(x)) and
               element (i,j) = 1 if string_arr[i] == np.unique(x)[j], 0 otherwise
             - classes is a 1d array of classes corresponding to the columns of onehot
    """

    classes, inverse = np.unique(x, return_inverse=True)
    onehot = np.eye(classes.shape[0], dtype='int64')[inverse]
    return (onehot, classes)


def cross_entropy(Yhat, Y):
    """
    Calculate row-wise cross entropy

    :param Yhat: NxK array where (i,j) gives the predicted probability of class j for sample i
    :param Y: either:
              1) NxK array where (i,j) gives the true probability of class j for sample i or
              2) a 1-D array where element i gives the index of the true class for sample i
    :return: 1-D array with N elements, where element i gives the cross entropy for the ith sample
    """

    if Y.ndim == 1:
        ce = -np.log(Yhat[np.arange(len(Y)), Y])
    else:
        ce = -np.sum(Y * np.log(Yhat), axis=1)

    return ce


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


class NNet():
    """
    NNet with support for multiple hidden layers
    """

    def __init__(self, Ws=None, y_classes=None):
        """
        Initialization

        :param Ws: optional list of weight matrices (list of 2-D numpy arrays)
        :param y_classes: optional array of y_classes (1-D numpy array with >= 2 elements)
        """

        self.Ws = Ws
        self.y_classes = y_classes

    def fit(self, X, y, hiddenNodes, stepSize=0.01, ITERS=100, seed=None):
        """
        Find the best weights via gradient descent

        :param X: training features
        :param y: training labels. 1-d array with >= 2 classes
        :param hiddenNodes: list indicating how many nodes to use in each hidden layer, excluding bias nodes
        :param stepSize: AKA "learning rate" AKA "alpha" used in gradient descent
        :param ITERS: How many gradient descent steps to make?
        :return: None. Update self.y_classes, self.Ws
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Validate hiddenNodes type
        if not isinstance(hiddenNodes, list):
            AssertionError("hiddenNodes should be a list of integers")

        # Determine unique y classes
        y01, y_classes = one_hot(y)
        if len(y_classes) < 2:
            AssertionError(f"y should have at least 2 distinct classes, but instead it has {len(y_classes)}")

        pass

    def predict(self, X, type='classes'):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :return: if type = 'probs' then probabilities else if type = 'classes' then classes
        """

        if self.Ws is None:
            raise AssertionError(f"Need to fit() before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.Ws[0]) - 1:
            raise AssertionError(f"Perceptron was fit on X with {len(self.Ws[0]) - 1} columns but this X has {X.shape[1]} columns")

        # Make predictions (forward pass)
        pass

#=== Test ============================================================================================

# Load simple images data
train = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_test.csv")

# Initialize & fit neural network
nn = NNet()
nn.fit(
    X = train.drop(columns='label').to_numpy(),
    y = train.label.to_numpy(),
    hiddenNodes = [5,3,4],
    stepSize = 0.3,
    ITERS = 10_000,
    seed = 0
)

# Evaluate on test data
preds = nn.predict(X = test.drop(columns='label').to_numpy())
(preds == test.label).mean()
