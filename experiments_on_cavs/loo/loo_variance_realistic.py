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

# Apply PCA to reduce to 2 dimensions
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# Function to compute the LOO estimate for a specified sample
def newton_estimate_loo(orig_model, X, y, j):
    ps = orig_model.predict_proba(X)
    W = np.diag(ps[:, 1] * (1 - ps[:, 1]))
    M = np.linalg.inv(X.T @ W @ X)
    y_ = 2 * y - 1
    p_j = ps[j, 1]
    X_j = X[j].reshape(1, -1)
    num = M @ X_j.T * (1 - p_j) * y_[j]
    denom = 1 - X_j @ M @ X_j.T * (1 - p_j) * p_j
    loo_estimate = num / denom
    return loo_estimate.flatten()

# Step 1: Train Logistic Regression model on the full dataset
logistic_model_full = LogisticRegression(max_iter=1000)
logistic_model_full.fit(X_pca_2, y)

# Plot the full data decision boundary
fig, ax = plt.subplots()
ax.scatter(X_pca_2[y == 0, 0], X_pca_2[y == 0, 1], color='blue', label='random500_1', alpha=0.5)
ax.scatter(X_pca_2[y == 1, 0], X_pca_2[y == 1, 1], color='red', label='anchor_concept1', alpha=0.5)

# Create mesh grid for decision boundary plot
x_min, x_max = X_pca_2[:, 0].min() - 1, X_pca_2[:, 0].max() + 1
y_min, y_max = X_pca_2[:, 1].min() - 1, X_pca_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Plot the decision boundary for the full model
Z_full = logistic_model_full.decision_function(grid).reshape(xx.shape)
ax.contour(xx, yy, Z_full, levels=[0], linewidths=2, colors='blue', label='Full Data Boundary')

# Step 2: Plot decision boundaries after leaving out each random point
for j in range(len(act_concept), len(X)):
    X_leave_out = np.delete(X_pca_2, j, axis=0)
    y_leave_out = np.delete(y, j)
    
    logistic_model_leave_out = LogisticRegression(max_iter=10000)
    logistic_model_leave_out.fit(X_leave_out, y_leave_out)
    
    Z_leave_out = logistic_model_leave_out.decision_function(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z_leave_out, levels=[0], linewidths=1, colors='black', linestyles='dotted', alpha=0.7)

# Step 3: Compute LOO scores and plot the decision boundaries using LOO scores
for j in range(len(act_concept), len(X)):
    loo_score = newton_estimate_loo(logistic_model_full, X_pca_2, y, j)

    # Calculate the slope of the LOO decision boundary
    if loo_score[1] != 0:
        loo_slope = -loo_score[0] / loo_score[1]
    else:
        loo_slope = np.inf

    # Calculate y-intercepts for plotting the LOO lines
    x_vals = np.linspace(x_min, x_max, 100)
    
    if loo_slope != np.inf:
        y_vals = loo_slope * x_vals
    else:
        y_vals = np.full_like(x_vals, np.mean(X_pca_2[:, 1]))

    # Ensure the line is within the plot limits
    y_vals = np.clip(y_vals, y_min, y_max)
    
    ax.plot(x_vals, y_vals, color='red', linestyle='dotted', alpha=0.7)

# Finalize plot
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Logistic Regression Decision Boundaries with LOO Adjustments')
ax.legend()
plt.show()