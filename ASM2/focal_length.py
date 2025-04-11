import numpy as np

# Load the file
data = np.load('stereo_calib_data.npz')

# Get the intrinsic matrices
mtxL = data['mtxL']
mtxR = data['mtxR']

# Focal length is usually the value at position (0, 0)
focal_length_left = mtxL[0, 0]
focal_length_right = mtxR[0, 0]

print("Focal length (left camera):", focal_length_left)
print("Focal length (right camera):", focal_length_right)
