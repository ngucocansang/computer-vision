import cv2
import numpy as np

# Define the checkerboard pattern size
CHECKERBOARD = (6, 8)  # (rows, columns)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# Lists to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save frames with detected corners, 'r' to run calibration, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the corners on the image
        cv2.drawChessboardCorners(frame, CHECKERBOARD, refined_corners, ret)
        cv2.imshow('Calibration Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s') and ret:  # Save valid frames
        objpoints.append(objp)
        imgpoints.append(refined_corners)
        cv2.imwrite(f'calib_frame_{len(imgpoints)}.jpg', frame)
        print(f"✅ Saved frame {len(imgpoints)}")
    elif key == ord('r'):  # Run calibration
        if len(objpoints) >= 5:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Print calibration results
            print("\n🎯 Camera Calibration Results:")
            print(f"Camera Matrix:\n{mtx}")
            print(f"Distortion Coefficients:\n{dist}")
            print(f"Rotation Vectors:\n{rvecs}")
            print(f"Translation Vectors:\n{tvecs}")
        else:
            print("\n❌ Not enough valid frames for calibration. Try again.")

cap.release()
cv2.destroyAllWindows()
