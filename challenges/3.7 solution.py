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

        if yval is not None:
            y01_val, yval_classes = one_hot(yval)

        # Initialization (note Ws is a list of weight matrices)
        gen = np.random.default_rng(seed)
        X1 = np.insert(X / 255, obj=X.shape[1], values=1, axis=1)
        Ws = [None] * (len(hiddenNodes) + 1)
        Ws[0] = gen.uniform(low=-1, high=1, size=(X1.shape[1], hiddenNodes[0]))
        for i in range(1, len(hiddenNodes)):
            Ws[i] = gen.uniform(low=-1, high=1, size=(hiddenNodes[i - 1] + 1, hiddenNodes[i]))
        Ws[i + 1] = gen.uniform(low=-1, high=1, size=(hiddenNodes[i] + 1, len(y_classes)))

        # Initialize lists to store Xs, Zs, and gradients
        Zs = [None] * len(Ws)
        Xs = [None] * len(Ws)
        gradWs = [None] * len(Ws)

        # Determine number of batches
        if batchSize is None:
            Nbatches = 1
        else:
            Nbatches = np.ceil(X1.shape[0]/batchSize).astype('int64')

        # Initialize lists to store performance stats
        CE_train_list = []
        Accuracy_train_list = []
        CE_validation_list = []
        Accuracy_validation_list = []

        # Train
        for i in range(ITERS):

            # mini batches
            idxs = gen.choice(X1.shape[0], size=X1.shape[0], replace=False)
            batches = np.array_split(idxs, Nbatches)

            # Loop over batches
            for b in range(Nbatches):
                batch_idxs = batches[b]
                Xs[0] = X1[batch_idxs]

                # Make predictions (forward pass)
                for j in range(len(Ws)):
                    Zs[j] = Xs[j] @ Ws[j]
                    if j + 1 < len(Xs):
                        Xs[j + 1] = np.insert(logistic(Zs[j]), obj=Zs[j].shape[1], values=1, axis=1)
                yhat_probs = softmax(Zs[-1])

                if b == Nbatches - 1:
                    # Calculate training cross entropy loss, accuracy
                    yhat_probs_train = self.predict(X, type='probs', Ws=Ws, y_classes=y_classes)
                    yhat_classes_train = y_classes[np.argmax(yhat_probs_train, axis=1)]
                    ce_train = cross_entropy(yhat_probs_train, y01)
                    CE_train = np.mean(ce_train)
                    accuracy_train = np.mean(yhat_classes_train == y)

                    CE_train_list.append(CE_train)
                    Accuracy_train_list.append(accuracy_train)

                    if Xval is None or yval is None:
                        print(f'iteration: {i}, train cross entropy loss: {CE_train}, train accuracy: {accuracy_train}')
                    else:
                        yhat_probs_val = self.predict(Xval, type='probs', Ws = Ws, y_classes = y_classes)
                        yhat_classes_val = y_classes[np.argmax(yhat_probs_val, axis=1)]
                        ce_val = cross_entropy(yhat_probs_val, y01_val)
                        CE_val = np.mean(ce_val)
                        accuracy_val = np.mean(yhat_classes_val == yval)

                        CE_validation_list.append(CE_val)
                        Accuracy_validation_list.append(accuracy_val)

                        print(f'iteration: {i}, '
                              f'train cross entropy loss: {CE_train}, validation cross entropy loss: {CE_val}, '
                              f'train accuracy: {accuracy_train}, validation accuracy: {accuracy_val}')


                # Calculate gradients (backward pass)
                gradZ = (yhat_probs - y01[batch_idxs])[:, None, :]
                for j in range(len(Ws) - 1, -1, -1):
                    gradWs[j] = np.transpose(Xs[j][:, None, :], axes=[0, 2, 1]) @ gradZ
                    gradWs[j] = gradWs[j].mean(axis=0)
                    gradX = (gradZ @ np.transpose(Ws[j]))[:, :, :-1]
                    gradZ = gradX * (Xs[j] * (1 - Xs[j]))[:, None, :-1]

                # Update weights (gradient step)
                for j in range(len(Ws)):
                    Ws[j] -= gradWs[j] * stepSize

        # Update class vars
        self.y_classes = y_classes
        self.Ws = Ws
        self.CE_train_list = CE_train_list
        self.Accuracy_train_list = Accuracy_train_list
        self.CE_validation_list = CE_validation_list
        self.Accuracy_validation_list = Accuracy_validation_list

    def predict(self, X, type='classes', Ws = None, y_classes = None):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :param type: If 'classes', predicted classes, else if 'probs', predicted class probabilities
        :param Ws: list of 2-D arrays (weight matrices). If None, use self.Ws
        :param y_classes: numpy array of y classes. If None, use self.y_classes
        :return: if type = 'probs' then probabilities else if type = 'classes' then classes
        """

        if Ws is None:
            Ws = self.Ws
        if y_classes is None:
            y_classes = self.y_classes

        if Ws is None:
            raise AssertionError(f"Need to fit() before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(Ws[0]) - 1:
            raise AssertionError(
                f"Perceptron was fit on X with {len(Ws[0]) - 1} columns but this X has {X.shape[1]} columns")

        # Make predictions (forward pass)
        X1 = np.insert(X / 255, obj=X.shape[1], values=1, axis=1)
        for j in range(len(Ws)):
            Z = X1 @ Ws[j]
            if j < len(Ws) - 1:
                X1 = np.insert(logistic(Z), obj=Z.shape[1], values=1, axis=1)
        yhat_probs = softmax(Z)

        if type == 'probs':
            return yhat_probs
        elif type == 'classes':
            yhat_classes = y_classes[np.argmax(yhat_probs, axis=1)]
            return yhat_classes

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