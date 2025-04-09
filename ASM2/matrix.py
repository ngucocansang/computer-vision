import csv
import statistics
import numpy as np

x_values = []
y_values = []

# Ensure this file exists in the same folder
with open('pupil_positions.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for row in reader:
        if len(row) >= 2:
            try:
                x_values.append(float(row[0]))
                y_values.append(float(row[1]))
            except ValueError:
                continue

if x_values and y_values:
    var_x = statistics.variance(x_values)
    var_y = statistics.variance(y_values)

    R = np.array([[var_x, 0],
                  [0, var_y]])

    print(f"Error Variance:\n  x: {var_x:.2f}, y: {var_y:.2f}")
    print("\nMeasurement Noise Covariance Matrix R:")
    print(R)
else:
    print("No data found in pupil_positions.csv.")
