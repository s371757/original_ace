import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the text files
root_directory = '/Users/juliawenkmann/Documents/original_ace'

# Initialize lists to hold the x and y values for the plot
x_values = []
y_values = []

# Loop through the text files and calculate the mean mean_variance for each file
for number in range(10, 55, 5):
    file_path = os.path.join(root_directory, f'variation_measures_{number}.txt')
    print(f"Processing file: {file_path}")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue
    
    mean_variances = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print(f"Contents of {file_path}:")
        for line in lines:
            print(line.strip())
            match = re.search(r'mean_variance\s*=\s*([0-9.]+)', line)
            if match:
                mean_variance = float(match.group(1))
                mean_variances.append(mean_variance)
    
    if mean_variances:
        mean_of_means = np.mean(mean_variances)
        x_values.append(number)
        y_values.append(mean_of_means)
        print(f"File {file_path} processed: mean_of_means={mean_of_means}")
    else:
        print(f"No valid data found in file: {file_path}")

# Fit the line to the log-log data
coeffs = np.polyfit(np.log(x_values), np.log(y_values), deg=1)

# Extract the slope (the first coefficient)
slope = coeffs[0]
print(f"The slope of the line on the log-log plot is approximately: {slope}")
print("As x increases, y decreases more rapidly than 1/x (closer to y \propto x^{-1.41}).")

# Plot the data
plt.figure()
plt.loglog(x_values, y_values, marker='o')
plt.xlabel('Number')
plt.ylabel('Mean of Mean Variances')
plt.title('Mean of Mean Variances Across All Concepts')
plt.grid(True)
plt.show()