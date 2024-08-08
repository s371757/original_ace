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

# Step 1: Apply PCA to reduce to 15 dimensions
pca_15 = PCA(n_components=15)
X_pca_15 = pca_15.fit_transform(X)

# Step 2: Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_pca_15, y)

# Function to compute the LOO estimate for a specified sample
def newton_estimate_loo(orig_model, X, y, j):
    N_SAMPLES = len(y)
    
    # Predicted probabilities for the original model
    ps = orig_model.predict_proba(X)
    
    # Create a diagonal matrix of weights
    W = np.diag(ps[:, 1] * (1 - ps[:, 1]))
    
    # Compute the inverse Hessian matrix
    M = np.linalg.inv(X.T @ W @ X)
    
    # Translate y from {0, 1} to {-1, 1}
    y_ = 2 * y - 1
    
    # Probability for the j-th sample
    p_j = ps[j, 1]
    
    # Compute the numerator and denominator for the LOO estimate
    X_j = X[j].reshape(1, -1)  # Ensure X_j is a row vector
    
    num = M @ X_j.T * (1 - p_j) * y_[j]
    denom = 1 - X_j @ M @ X_j.T * (1 - p_j) * p_j
    
    # LOO estimate
    loo_estimate = num / denom
    
    return loo_estimate.flatten()

# Compute LOO scores for the random set
random_set_indices = range(len(act_concept), len(X))  # Indices of the random set in X
loo_scores = []
for j in random_set_indices:
    loo_score = newton_estimate_loo(logistic_model, X_pca_15, y, j)
    loo_scores.append(np.abs(loo_score))  # Compute the norm (absolute value) of the LOO score
    print(j)
    print(loo_score)


vector_magnitudes = [np.linalg.norm(vector) for vector in loo_scores]
# Calculate the mean length of the vectors
mean_length = np.mean(vector_magnitudes)

print("Mean length: ")
print(mean_length)
