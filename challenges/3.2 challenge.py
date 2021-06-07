import numpy as np
import pandas as pd

#=== Challenge ============================================================================================

class NNet():
    """
    My first neural network
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

    def fit(self, X, y, hiddenNodes, GUESSES=10_000, seed=None):
        """
        Randomly guess weights. Keep the best

        :param X: training features
        :param y: training labels. Should have exactly 2 classes
        :param hiddenNodes: How many hidden layer nodes to use, excluding bias node
        :param GUESSES: How many times to guess random weights
        :param seed: Optional random number seed
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

    def predict(self, X):
        """
        Predict on X

        :param X: 2-D array with >= 1 column of real-valued features
        :return: 1-D array of predicted values
        """

        pass

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
    GUESSES = 10_000,
    seed = 123
)

# Evaluate on test data
preds = nn.predict(X = test.drop(columns='label').to_numpy())
(preds == (test.label == 'checkerboard')).mean()
