import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to load .npy data from specified files
def load_data(file1, file2):
    data = []
    labels = []
    for file, label in [(file1, 0), (file2, 1)]:
        activations = np.load(file)
        # Flatten the data
        activations = activations.reshape(activations.shape[0], -1)
        data.append(activations)
        labels.extend([label] * len(activations))
    data = np.vstack(data)
    return data, labels

# Files containing the data
file1 = 'experiments_on_cavs/acts/random500_3.npy'
file2 = 'experiments_on_cavs/acts/anchor_concept1.npy'

# Load and preprocess data
data, labels = load_data(file1, file2)
X = np.array(data)
y = np.array(labels)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the transformed data
plt.figure()
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='random500_1', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='anchor_concept1', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformation to 2D')
plt.legend()
plt.show()