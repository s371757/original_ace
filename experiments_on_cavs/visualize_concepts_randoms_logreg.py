import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression, LinearRegression
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

# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_pca, y)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_pca, y)

# Plot the transformed data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], color='blue', label='random500_1', alpha=0.5)
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], color='red', label='anchor_concept1', alpha=0.5)

# Create a mesh grid for plotting decision boundary
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
z_min, z_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
zz = -(logistic_model.coef_[0][0] * xx + logistic_model.coef_[0][1] * yy + logistic_model.intercept_[0]) / logistic_model.coef_[0][2]

# Logistic Regression decision boundary
ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

# Linear Regression decision plane
zz_linear = -(linear_model.coef_[0] * xx + linear_model.coef_[1] * yy + linear_model.intercept_) / linear_model.coef_[2]
ax.plot_surface(xx, yy, zz_linear, color='yellow', alpha=0.3)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA with Logistic and Linear Regression Decision Boundaries')
ax.legend()
plt.show()