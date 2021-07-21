import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, linewidth=999)

# Load the MNIST dataset
mnist_train = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_train.csv')
mnist_train.head()

# Inspect its shape
mnist_train.shape

# Print the data for the first image
mnist_train.iloc[0, 1:].to_numpy().reshape(28,28)

# Plot the first image
import matplotlib.pyplot as plt
plt.imshow(mnist_train.iloc[0, 1:].to_numpy().reshape(28,28), cmap='gray', vmin=0, vmax=255)

# Check its label
mnist_train.loc[0, 'label']