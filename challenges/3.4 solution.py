import numpy as np
import pandas as pd

#=== Challenge ============================================================================================

def softmax(x):
    """
    Calculate row-wise softmax

    :param x: 2d array
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
    NNet with multiclass support
    """

    def __init__(self, W1=None, W2=None, y_classes=None):
        """
        Initialization

        :param W1: optional weight matrix, W1 (2-D numpy array)
        :param W2: optional weight matrix, W2 (2-D numpy array)
        :param y_classes: optional array of y_classes (1-D numpy array with >= 2 elements)
        """
        self.W1 = W1
        self.W2 = W2
        self.y_classes = y_classes

    def fit(self, X, y, hiddenNodes, stepSize=0.01, ITERS=100, seed=None):
        """
        Find the best weights via gradient descent

        :param X: training features
        :param y: training labels
        :param hiddenNodes: How many hidden layer nodes to use, excluding bias node
        :param stepSize: AKA "learning rate" AKA "alpha" used in gradient descent
        :param ITERS: How many gradient descent steps to make?
        :return: None. Update self.y_classes, self.W1, self.W2
        """

        # Validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        # Determine unique y classes
        y01, y_classes = one_hot(y)
        if len(y_classes) < 2:
            AssertionError(f"y should have 2 at least 2 distinct classes, but instead it has {len(y_classes)}")

        # Initialization
        gen = np.random.default_rng(seed)
        X1 = np.insert(X / 255, obj=X.shape[1], values=1, axis=1)
        W1 = gen.uniform(low=-1, high=1, size=(X.shape[1] + 1, hiddenNodes))
        W2 = gen.uniform(low=-1, high=1, size=(W1.shape[1] + 1, len(y_classes)))

        # Gradient Descent
        for i in range(ITERS):

            # Make predictions (forward pass)
            Z1 = X1 @ W1
            X2 = np.insert(logistic(Z1), obj=Z1.shape[1], values=1, axis=1)
            Z2 = X2 @ W2
            yhat_probs = softmax(Z2)
            yhat_classes = y_classes[np.argmax(yhat_probs, axis=1)]

            # Calculate cross entropy loss, accuracy
            ce = cross_entropy(yhat_probs, y01)
            CE = np.mean(ce)
            accuracy = np.mean(yhat_classes == y)
            if i % 100 == 0:
                print(f'iteration: {i}, cross entropy loss: {CE}, accuracy: {accuracy}')

            # Calculate gradients                                           # Dimensionality
            gradZ2 = (yhat_probs - y01)[:, None, :]                         # (4000, 1, 3)
            gradW2 = np.transpose(X2[:, None, :], axes=[0, 2, 1]) @ gradZ2  # (4000, 5, 3)
            gradW2 = gradW2.mean(axis=0)                                    # (5, 3)
            gradX2 = (gradZ2 @ np.transpose(W2))[:, :, :-1]                 # (4000, 1, 4)
            gradZ1 = gradX2 * (X2 * (1 - X2))[:, None, :-1]                 # (4000, 1, 4)
            gradW1 = np.transpose(X1[:, None, :], axes=[0, 2, 1]) @ gradZ1  # (4000, 5, 4)
            gradW1 = gradW1.mean(axis=0)                                    # (5, 4)

            # Update weights
            W1 -= gradW1 * stepSize
            W2 -= gradW2 * stepSize

        # Update class vars
        self.y_classes = y_classes
        self.W1 = W1
        self.W2 = W2

    def predict(self, X, type='classes'):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :return: if type = 'probs' then probabilities else if type = 'classes' then classes
        """

        if self.W1 is None:
            raise AssertionError(f"Need to fit() before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.W1) - 1:
            raise AssertionError(f"Perceptron was fit on X with {len(self.W1) - 1} columns but this X has {X.shape[1]} columns")

        X1 = np.insert(X / 255, obj=X.shape[1], values=1, axis=1)
        Z1 = X1 @ self.W1
        X2 = np.insert(logistic(Z1), obj=Z1.shape[1], values=1, axis=1)
        Z2 = X2 @ self.W2
        yhat_probs = softmax(Z2)

        if type == 'probs':
            return yhat_probs
        elif type == 'classes':
            yhat_classes = self.y_classes[np.argmax(yhat_probs, axis=1)]
            return yhat_classes

#=== Test ============================================================================================

# Load simple images data
train = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_test.csv")

# Initialize & fit neural network
nn = NNet()
nn.fit(
    X = train.drop(columns='label').to_numpy(),
    y = train.label.to_numpy(),
    hiddenNodes = 4,
    stepSize = 0.3,
    ITERS = 10_000,
    seed = 0
)

# Evaluate on test data
preds = nn.predict(X = test.drop(columns='label').to_numpy())
(preds == test.label).mean()
