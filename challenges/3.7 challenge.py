import numpy as np
import pandas as pd

#=== Bonus Challenge ============================================================================================
# Add validation monitoring to the network you implemented in challenge 3.6
# The fit method should accept Xval, and yval, and print the progress of both train and validation accuracy and crossentropy

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
    NNet with stochastic gradient descent and validation loss monitoring
    """

    def __init__(self, Ws=None, y_classes=None):
        """
        Initialization

        :param Ws: optional list of weight matrices (list of 2-D numpy arrays)
        :param y_classes: optional array of y_classes (1-D numpy array with >= 2 elements)
        """

        self.Ws = Ws
        self.y_classes = y_classes

    def fit(self, X, y, hiddenNodes, Xval=None, yval=None, stepSize=0.01, ITERS=100, batchSize=None, seed=None):
        """
        Find the best weights via stochastic gradient descent

        :param X: training features
        :param y: training labels. 1-d array with >= 2 classes
        :param hiddenNodes: list indicating how many nodes to use in each hidden layer, excluding bias nodes
        :param Xval: optional validation features
        :param yval: optional validation labels. 1-d array with >= 2 classes
        :param stepSize: AKA "learning rate" AKA "alpha" used in gradient descent
        :param ITERS: How many gradient descent steps to make?
        :param batchSize: How many samples to user per batch? If None, use all samples
        :return: None. Update self.y_classes, self.W1, self.W2
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Validate Ws type
        if not isinstance(hiddenNodes, list):
            AssertionError("hiddenNodes should be a list of integers")

        # Determine unique y classes
        y01, y_classes = one_hot(y)
        if len(y_classes) < 2:
            AssertionError(f"y should have at least 2 distinct classes, but instead it has {len(y_classes)}")

        pass

    def predict(self, X, type='classes', Ws = None, y_classes = None):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :param type: If 'classes', predicted classes, else if 'probs', predicted class probabilities
        :param Ws: list of 2-D arrays (weight matrices). If None, use self.Ws
        :param y_classes: numpy array of y classes. If None, use self.y_classes
        :return: if type = 'probs' then probabilities else if type = 'classes' then classes
        """

        pass

#=== Test ============================================================================================

# Load MNIST images data
mnist_train = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_train.csv")
mnist_test = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_test.csv")

# Split train into train and validation
gen = np.random.default_rng(seed = 1234)
val_ids, train_ids = np.split(gen.choice(len(mnist_train), len(mnist_train), replace=False), [5000])

# Initialize & fit neural network
nn = NNet()
nn.fit(
    X = mnist_train.iloc[train_ids].drop(columns='label').to_numpy(),
    y = mnist_train.iloc[train_ids].label.to_numpy(),
    Xval = mnist_train.iloc[val_ids].drop(columns='label').to_numpy(),
    yval = mnist_train.iloc[val_ids].label.to_numpy(),
    hiddenNodes = [50, 20, 8],
    stepSize = 0.1,
    batchSize = 100,
    ITERS = 200,
    seed = 0
)