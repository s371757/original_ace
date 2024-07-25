import os
import pickle
import numpy as np
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_cav(file_path):
    with open(file_path, 'rb') as f:
        cav = pickle.load(f)
    return cav

def calculate_variance_of_cavs(cav_list):
    cav_matrix = np.array(cav_list)
    variances = np.var(cav_matrix, axis=0)
    return variances

def process_file(file_path):
    cav_data = load_cav(file_path)
    cav_array = cav_data['cavs'][0]  # Take the first CAV array
    concept = os.path.basename(file_path).split('-')[0]
    return concept, cav_array

def process_cavs(directory):
    concept_cavs = {}
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl') and 'anchor_concept' in f]
    file_count = len(file_list)
    print(f"Total number of files to process: {file_count}")

    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in file_list}
        for idx, future in enumerate(as_completed(futures)):
            concept, cav_array = future.result()
            if concept not in concept_cavs:
                concept_cavs[concept] = []
            concept_cavs[concept].append(cav_array)
            if (idx + 1) % 10 == 0 or idx == file_count - 1:
                print(f"Processed {idx + 1}/{file_count} files")

    end_time = time.time()
    print(f"Time taken to group CAVs by concept: {end_time - start_time:.2f} seconds")
    
    return concept_cavs

def write_variances_to_file(concept_cavs, output_file):
    start_time = time.time()

    with open(output_file, 'w') as f:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(calculate_variance_of_cavs, cav_list): concept for concept, cav_list in concept_cavs.items()}
            for future in as_completed(futures):
                concept = futures[future]
                variances = future.result()
                variance_str = ", ".join(f"{var:.4f}" for var in variances)
                f.write(f"{concept}: {variance_str}\n")
                print(f"Processed {concept} with variances {variance_str}")
                gc.collect()  # Garbage collect to free up memory

    end_time = time.time()
    print(f"Time taken to write variances to file: {end_time - start_time:.2f} seconds")

def main():
    # Directory containing the CAVs
    directory = "/Users/juliawenkmann/Documents/original_ace/results/cavs"
    # Output file to write the variances
    output_file = "variance_of_cavs.txt"

    # Process the CAVs and calculate the variances
    concept_cavs = process_cavs(directory)
    # Write the results to a file
    write_variances_to_file(concept_cavs, output_file)

    print("Variance of CAVs written to", output_file)

import numpy as np

def read_variances_from_file(file_path):
    concept_variances = {}
    with open(file_path, 'r') as f:
        for line in f:
            concept, variances_str = line.strip().split(':')
            variances = np.array([float(v) for v in variances_str.split(',')])
            concept_variances[concept] = variances
    return concept_variances

def calculate_variation_measures(concept_variances):
    variation_measures = {}
    for concept, variances in concept_variances.items():
        mean_variance = np.mean(variances)
        std_deviation = np.std(variances)
        variation_measures[concept] = {
            'mean_variance': mean_variance,
            'std_deviation': std_deviation
        }
    return variation_measures

def write_variation_measures_to_file(variation_measures, output_file):
    with open(output_file, 'w') as f:
        for concept, measures in variation_measures.items():
            f.write(f"{concept}: mean_variance={measures['mean_variance']:.4f}, std_deviation={measures['std_deviation']:.4f}\n")

def main_processing():
    # Input file containing the variances
    input_file = "variance_of_cavs.txt"
    # Output file to write the variation measures
    output_file = "variation_measures.txt"

    # Read variances from the input file
    concept_variances = read_variances_from_file(input_file)
    # Calculate variation measures
    variation_measures = calculate_variation_measures(concept_variances)
    # Write the variation measures to a file
    write_variation_measures_to_file(variation_measures, output_file)

    print("Variation measures written to", output_file)

if __name__ == "__main__":
    main()
    main_processing()