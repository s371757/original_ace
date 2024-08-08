import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Function to load .npy data from specified files
def load_data(file):
    activations = np.load(file)
    return activations.reshape(activations.shape[0], -1)

# Load activation vectors from the specified folders
file1 = 'experiments_on_cavs/acts/random500_1.npy'
file2 = 'experiments_on_cavs/acts/anchor_concept1.npy'

act_random = load_data(file1)
act_concept = load_data(file2)

# Combine data and create labels
X = np.vstack([act_concept, act_random])
y = np.array([1] * len(act_concept) + [0] * len(act_random))

# Step 1: Apply PCA to reduce to 2 dimensions
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# Step 2: Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_pca_2, y)

# Step 3: Plot the transformed data
fig, ax = plt.subplots()
ax.scatter(X_pca_2[y == 0, 0], X_pca_2[y == 0, 1], color='blue', label='random500_1', alpha=0.5)
ax.scatter(X_pca_2[y == 1, 0], X_pca_2[y == 1, 1], color='red', label='anchor_concept1', alpha=0.5)

# Create a finer mesh grid for plotting decision boundary
x_min, x_max = X_pca_2[:, 0].min() - 1, X_pca_2[:, 0].max() + 1
y_min, y_max = X_pca_2[:, 1].min() - 1, X_pca_2[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = logistic_model.decision_function(grid).reshape(xx.shape)

# Plot decision boundary
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA with Logistic Regression Decision Boundary')
ax.legend()
plt.show()