import os
import pickle
import numpy as np
import time
import gc

def load_cav(file_path):
    with open(file_path, 'rb') as f:
        cav = pickle.load(f)
    return cav

def calculate_trace_of_covariance(cav_list):
    cav_matrix = np.array(cav_list)
    print(f"calculated cav_matrix with shape {cav_matrix.shape}")
    cov_matrix = np.cov(cav_matrix, rowvar=False)
    print("calculated cov matrix")
    trace = np.trace(cov_matrix)
    return trace

def process_cavs(directory):
    concept_cavs = {}
    file_list = [f for f in os.listdir(directory) if f.endswith('.pkl') and 'anchor_concept' in f]
    file_count = len(file_list)
    print(f"Total number of files to process: {file_count}")

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

def write_traces_to_file(concept_cavs, output_file):
    start_time = time.time()

    with open(output_file, 'w') as f:
        for concept, cav_list in concept_cavs.items():
            trace = calculate_trace_of_covariance(cav_list)
            f.write(f"{concept}: {trace:.4f}\n")
            print(f"Processed {concept} with trace {trace:.4f}")
            gc.collect()  # Garbage collect to free up memory

    end_time = time.time()
    print(f"Time taken to write traces to file: {end_time - start_time:.2f} seconds")

# Directory containing the CAVs
directory = "C:/Users/Julia/Documents/Coding_Projects/BA/ACE/results/cavs"
# Output file to write the traces
output_file = "trace_of_covariance.txt"

# Process the CAVs and calculate the traces
concept_cavs = process_cavs(directory)
# Write the results to a file
write_traces_to_file(concept_cavs, output_file)

print("Trace of covariance matrices written to", output_file)
