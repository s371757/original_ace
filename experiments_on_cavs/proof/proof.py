import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Function to load .npy data from specified files
def load_data(file):
    activations = np.load(file)
    return activations.reshape(activations.shape[0], -1)

# Function to calculate the decision function ψ_θ(X)
def psi_theta(X, theta):
    return np.dot(X, theta)

# Function to calculate the Hessian matrix for logistic regression
def calculate_hessian(X, probs):
    W = np.diag(probs * (1 - probs))
    H = X.T @ W @ X
    return H

# Function to create a mixed distribution of concept and random examples
def get_mixed_distribution(concept_examples, random_examples, ratio, num_samples):
    mixed_X, mixed_y = [], []
    
    for _ in range(num_samples):
        if np.random.random() <= ratio:
            idx = np.random.randint(len(concept_examples))
            mixed_X.append(concept_examples[idx])
            mixed_y.append(1)  # Label for concept
        else:
            idx = np.random.randint(len(random_examples))
            mixed_X.append(random_examples[idx])
            mixed_y.append(0)  # Label for random
    
    return np.array(mixed_X), np.array(mixed_y)

# Main analysis steps

# 1. Load the concept data
file2 = 'experiments_on_cavs/acts/anchor_concept1.npy'
act_concept = load_data(file2)

# Initialize a list to store the differences sqrt(n) * (θ_n - θ_0)
theta_diffs = []

# Variable to store the reference logistic regression parameters θ_0
theta_0 = None

# 2. Loop over different random sets and perform logistic regression
for i in range(1, 50):
    # Load random data for the current iteration
    file1 = f'experiments_on_cavs/acts/random500_{i}.npy'
    act_random = load_data(file1)

    # Combine concept and random data and create labels
    X = np.vstack([act_concept, act_random])
    y = np.array([1] * len(act_concept) + [0] * len(act_random))

    # Fit logistic regression model to obtain θ_n
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X, y)
    theta_n = np.concatenate([logistic_model.intercept_, logistic_model.coef_.flatten()])

    if i == 1:
        # Save the first model's parameters as θ_0
        theta_0 = theta_n
    else:
        # Calculate sqrt(n) * (θ_n - θ_0) and store the result
        n = len(X)
        theta_diff = np.sqrt(n) * (theta_n - theta_0)
        theta_diffs.append(theta_diff)

        # Compute ψ_θ0(X) for all Xi using θ_0
        psi_theta0_X = psi_theta(X, theta_0[1:]) + theta_0[0]

        # Calculate 1/sqrt(n) * sum(ψ_θ0(Xi))
        psi_theta0_sum = np.sum(psi_theta0_X) / np.sqrt(n)

        # 3. Estimate E(ψ_θ0) using Monte Carlo simulation
        m = 10000  # Number of Monte Carlo samples
        psi_theta0_monte_carlo_sum = 0

        for _ in range(m):
            # Generate a mixed distribution sample
            X_mixed, y_mixed = get_mixed_distribution(act_concept, act_random, ratio=0.5, num_samples=len(X))
            psi_theta0_mixed = psi_theta(X_mixed, theta_0[1:]) + theta_0[0]
            psi_theta0_monte_carlo_sum += np.mean(psi_theta0_mixed)

        E_psi_theta0 = psi_theta0_monte_carlo_sum / m

        # 4. Calculate the Hessian matrix and its inverse at θ_n
        probs = logistic_model.predict_proba(X)[:, 1]
        Hessian = calculate_hessian(X, probs)
        Hessian_inv = np.linalg.inv(Hessian)

        # 5. Calculate the right-hand side of the equation
        rhs = Hessian_inv @ psi_theta0_sum

        # 6. Compare sqrt(n) * (θ_n - θ_0) with the right-hand side
        difference = theta_diff - rhs

        # Calculate the norm of the difference to quantify the difference
        diff_norm = np.linalg.norm(difference)
        print(f"Iteration {i}: Norm of the difference = {diff_norm}")

# 7. Save sqrt(n) * (θ_n - θ_0) to a file for analysis
np.save('theta_diffs.npy', np.array(theta_diffs))
print("Saved theta differences to 'theta_diffs.npy'")

if __name__ == "__main__":
    # Run the analysis steps sequentially
    pass  # Replace with a function call if needed