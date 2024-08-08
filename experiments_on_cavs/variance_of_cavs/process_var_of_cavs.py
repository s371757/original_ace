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

def main():
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