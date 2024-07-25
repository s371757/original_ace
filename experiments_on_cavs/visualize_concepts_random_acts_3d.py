import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
file1 = 'experiments_on_cavs/acts/random500_1.npy'
file2 = 'experiments_on_cavs/acts/anchor_concept1.npy'

# Load and preprocess data
data, labels = load_data(file1, file2)
X = np.array(data)
y = np.array(labels)

# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Plot the transformed data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], color='blue', label='random500_1', alpha=0.5)
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], color='red', label='anchor_concept1', alpha=0.5)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Transformation to 3D')
ax.legend()

plt.show()