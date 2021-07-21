import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the simple images data
train = pd.read_csv('https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/simple_images_train.csv')
train.head()

# Inspect
train.shape

# Plot the first simple image
plt.imshow(train.drop(columns='label').to_numpy()[0].reshape(2,2), cmap='gray', vmin=0, vmax=255)