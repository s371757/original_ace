import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from ace_helpers import get_acts_from_images

# Load images and compute activations
def load_images(image_dir, max_images=None):
    """Load images from a directory."""
    image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    if max_images:
        image_files = image_files[:max_images]
    images = [np.array(Image.open(img)) for img in image_files]
    return images

def compute_activations(images, model, bottleneck_layer):
    """Compute activations for a list of images using a specific model layer."""
    activations = get_acts_from_images(images, model, bottleneck_layer)
    return activations

def main():
    # Define directories and parameters
    anchor_concept_dir = "/Users/juliawenkmann/Documents/original_ace/results/concepts/mixed4c_anchor_concept1_patches"
    random_images_dir = "/Users/juliawenkmann/Documents/original_ace/dataset/random500_0"
    model = ...  # Load your pre-trained model here
    bottleneck_layer = 'mixed4c'  # Adjust based on your model

    # Load images
    anchor_images = load_images(anchor_concept_dir)
    random_images = load_images(random_images_dir)

    # Compute activations
    anchor_activations = compute_activations(anchor_images, model, bottleneck_layer)
    random_activations = compute_activations(random_images, model, bottleneck_layer)

    # Combine activations
    all_activations = np.vstack((anchor_activations, random_activations))

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_activations)

    # Split PCA result back into the two sets
    anchor_pca = pca_result[:len(anchor_activations)]
    random_pca = pca_result[len(anchor_activations):]

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(anchor_pca[:, 0], anchor_pca[:, 1], label='Anchor Concept', color='blue')
    plt.scatter(random_pca[:, 0], random_pca[:, 1], label='Random Images', color='red')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.title('PCA of Activation Vectors')
    plt.show()

if __name__ == "__main__":
    main()