import os
import pickle
import numpy as np
import time
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_cav(file_path):
    with open(file_path, 'rb') as f:
        cav = pickle.load(f)
    return cav

def calculate_cosine_similarity(cav_list):
    cav_matrix = np.array(cav_list)
    similarity_matrix = cosine_similarity(cav_matrix)
    return similarity_matrix

def process_cavs(directory):
    concept_cavs = {}
    file_list = [f for f in os.listdir(directory) if f.endswith('.pkl') and 'anchor_concept' in f]
    file_count = len(file_list)

    start_time = time.time()
    
    for idx, file_name in enumerate(file_list):
        concept = file_name.split('-')[0]  # Example: 'anchor_concept19'
        file_path = os.path.join(directory, file_name)
        cav_data = load_cav(file_path)
        cav_array = cav_data['cavs'][0]  # Take the first CAV array
        
        if concept not in concept_cavs:
            concept_cavs[concept] = []
        
        concept_cavs[concept].append(cav_array)
        
        if (idx + 1) % 10 == 0 or idx == file_count - 1:
            print(f"Processed {idx + 1}/{file_count} files")

    end_time = time.time()
    print(f"Time taken to group CAVs by concept: {end_time - start_time:.2f} seconds")
    
    return concept_cavs

def write_similarities_to_file(concept_cavs, output_file):
    start_time = time.time()

    with open(output_file, 'w') as f:
        for concept, cav_list in concept_cavs.items():
            similarity_matrix = calculate_cosine_similarity(cav_list)
            avg_similarity = np.mean(similarity_matrix)
            f.write(f"{concept}: {avg_similarity:.4f}\n")
            print(f"Processed {concept} with average cosine similarity {avg_similarity:.4f}")
            gc.collect()  # Garbage collect to free up memory

    end_time = time.time()
    print(f"Time taken to write similarities to file: {end_time - start_time:.2f} seconds")

def visualize_pca(concept_cavs):
    all_cavs = []
    labels = []
    for concept, cav_list in concept_cavs.items():
        all_cavs.extend(cav_list)
        labels.extend([concept] * len(cav_list))
    
    pca = PCA(n_components=2)
    cavs_reduced = pca.fit_transform(all_cavs)
    
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(set(labels)):
        idxs = [j for j, x in enumerate(labels) if x == label]
        plt.scatter(cavs_reduced[idxs, 0], cavs_reduced[idxs, 1], label=label)
    
    plt.title('PCA of CAVs')
    plt.legend()
    plt.show()

def visualize_tsne(concept_cavs):
    all_cavs = []
    labels = []
    for concept, cav_list in concept_cavs.items():
        all_cavs.extend(cav_list)
        labels.extend([concept] * len(cav_list))
    
    tsne = TSNE(n_components=2, random_state=0)
    cavs_reduced = tsne.fit_transform(all_cavs)
    
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(set(labels)):
        idxs = [j for j, x in enumerate(labels) if x == label]
        plt.scatter(cavs_reduced[idxs, 0], cavs_reduced[idxs, 1], label=label)
    
    plt.title('t-SNE of CAVs')
    plt.legend()
    plt.show()

def plot_decision_boundary(model, X, y):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    h = .02  # step size in the mesh
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', marker='o')
    plt.title('Decision Boundary')
    plt.show()

def perform_clustering(concept_cavs, n_clusters=2):
    all_cavs = []
    labels = []
    for concept, cav_list in concept_cavs.items():
        all_cavs.extend(cav_list)
        labels.extend([concept] * len(cav_list))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(all_cavs)

    pca = PCA(n_components=2)
    cavs_reduced = pca.fit_transform(all_cavs)
    
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        idxs = np.where(clusters == i)
        plt.scatter(cavs_reduced[idxs, 0], cavs_reduced[idxs, 1], label=f'Cluster {i}')
    
    plt.title('K-Means Clustering of CAVs (PCA Reduced)')
    plt.legend()
    plt.show()

def main():
    # Directory containing the CAVs
    directory = "/Users/juliawenkmann/Documents/original_ace/results/cavs"
    # Output file to write the similarities
    output_file = "cosine_similarities.txt"

    # Process the CAVs and calculate the similarities
    concept_cavs = process_cavs(directory)
    # Write the results to a file
    write_similarities_to_file(concept_cavs, output_file)

    print("Cosine similarities written to", output_file)
    
    # Visualize with PCA
    visualize_pca(concept_cavs)
    
    # Visualize with t-SNE
    visualize_tsne(concept_cavs)
    
    # Train logistic regression model
    all_cavs = []
    labels = []
    for concept, cav_list in concept_cavs.items():
        all_cavs.extend(cav_list)
        labels.extend([concept] * len(cav_list))
    
    model = LogisticRegression()
    model.fit(all_cavs, labels)
    
    # Plot decision boundary
    plot_decision_boundary(model, np.array(all_cavs), np.array(labels))
    
    # Perform clustering
    perform_clustering(concept_cavs)

if __name__ == '__main__':
    main()