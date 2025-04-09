import pandas as pd

df = pd.read_csv('pupil_positions.csv')
var_x = df['x'].var()
var_y = df['y'].var()

print(f'Error Variance:\n  x: {var_x:.2f}, y: {var_y:.2f}')
