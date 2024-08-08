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
file1 = '/mnt/data/random500_1.npy'
file2 = '/mnt/data/anchor_concept1.npy'
act_random = load_data(file1)
act_concept = load_data(file2)

# Combine data and create labels
X = np.vstack([act_concept, act_random])
y = np.array([1] * len(act_concept) + [0] * len(act_random))

# Step 1: Apply PCA to reduce to 50 dimensions
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X)

# Step 2: Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_pca_50, y)

# Step 3: Compute the LOO estimate for a specified sample using the provided formula
def newton_estimate_loo(orig_model, X, y, j):
    # Predicted probabilities for the original model
    ps = orig_model.predict_proba(X)[:, 1]
    
    # Create a diagonal matrix of weights
    W = np.diag(ps * (1 - ps))
    
    # Compute the inverse Hessian matrix
    M = np.linalg.inv(X.T @ W @ X)
    
    # Translate y from {0, 1} to {-1, 1}
    y_ = 2 * y - 1
    
    # Probability for the j-th sample
    p_j = ps[j]
    
    # Compute the numerator and denominator for the LOO estimate
    X_j = X[j].reshape(1, -1)  # Ensure X_j is a row vector
    
    num = M @ X_j.T * (1 - p_j) * y_[j]
    denom = 1 - X_j @ M @ X_j.T * (1 - p_j) * p_j
    
    # LOO estimate
    loo_estimate = num / denom
    
    return loo_estimate.flatten()

# Specify the sample index for which to compute the LOO estimate
j = 0  # Change this to the index of the sample you are interested in

# Compute the LOO estimate for the specified sample
loo_estimate = newton_estimate_loo(logistic_model, X_pca_50, y, j)

# Step 4: Apply PCA to reduce to 2 dimensions for plotting
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_pca_50)
loo_pca_2 = pca_2.transform(loo_estimate.reshape(1, -1))

# Step 5: Plot the transformed data
fig, ax = plt.subplots()
ax.scatter(X_pca_2[y == 0, 0], X_pca_2[y == 0, 1], color='blue', label='random500_1', alpha=0.5)
ax.scatter(X_pca_2[y == 1, 0], X_pca_2[y == 1, 1], color='red', label='anchor_concept1', alpha=0.5)

# Plot the LOO estimate
ax.scatter(loo_pca_2[0, 0], loo_pca_2[0, 1], color='green', label=f'LOO sample {j}', alpha=0.8, edgecolor='black')

# Create a mesh grid for plotting decision boundary
x_min, x_max = X_pca_2[:, 0].min() - 1, X_pca_2[:, 0].max() + 1
y_min, y_max = X_pca_2[:, 1].min() - 1, X_pca_2[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
Z = logistic_model.predict(pca_50.inverse_transform(pca_2.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=1, alpha=0.5, colors=['green'])

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA with Logistic Regression Decision Boundary and LOO Estimate')
ax.legend()
plt.show()