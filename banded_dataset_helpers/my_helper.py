import os
import shutil
import random

# Source folder containing the pictures
source_folder = r'C:\Users\Julia\Documents\Coding_Projects\BA\BA\bachelorthesis\slic_experiments\shapes_experiments\dataset'

# Destination base folder where the new folders will be created
destination_base_folder = r'C:\Users\Julia\Documents\Coding_Projects\BA\ACE\shapes_dataset'

# Number of folders to create
num_folders = 50

# Number of pictures to sample in each folder
num_samples = 50

# Ensure the destination base folder exists
if not os.path.exists(destination_base_folder):
    os.makedirs(destination_base_folder)

# Get a list of all picture files in the source folder
all_pictures = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Check if there are enough pictures to sample
if len(all_pictures) < num_samples:
    raise ValueError(f"Not enough pictures in the source folder to sample {num_samples} pictures")

# Create folders and sample pictures
for i in range(1, num_folders + 1):
    folder_name = f'random500_{i}'
    destination_folder = os.path.join(destination_base_folder, folder_name)
    
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Randomly sample pictures
    sampled_pictures = random.sample(all_pictures, num_samples)
    
    # Copy sampled pictures to the destination folder
    for picture in sampled_pictures:
        src_path = os.path.join(source_folder, picture)
        dst_path = os.path.join(destination_folder, picture)
        shutil.copy(src_path, dst_path)

print("Folders created and pictures sampled successfully.")
