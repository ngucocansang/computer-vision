import cv2
import numpy as np
import time

def detect_pupil(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0][:2]  # return (x, y)
    return None

# Kalman Filter setup
dt = 1  # time step (can be tuned)
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

x = np.zeros((4, 1))           # Initial state
P = np.eye(4) * 1000           # Initial covariance
Q = np.eye(4) * 0.1            # Process noise
R = np.array([[8, 0], [0, 8]]) # Measurement noise (tune this based on your earlier output)

# For tracking accuracy
errors = []

# Open webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    measurement = detect_pupil(frame)

    # Kalman Predict
    x = F @ x
    P = F @ P @ F.T + Q

    if measurement is not None:
        z = np.array([[measurement[0]], [measurement[1]]])

        # Kalman Update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - H @ x
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        # Draw detection and compute error
        cv2.circle(frame, (int(z[0]), int(z[1])), 8, (0, 255, 0), 2)  # Green = detection
        prediction = (int(x[0]), int(x[1]))
        error = np.linalg.norm(z[:2] - x[:2])
        errors.append(error)

    # Draw prediction
    prediction = (int(x[0]), int(x[1]))
    cv2.circle(frame, prediction, 5, (0, 0, 255), -1)  # Red = Kalman prediction

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display
    cv2.imshow('Kalman Filter - Pupil Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Accuracy Report
if errors:
    avg_error = np.mean(errors)
    print(f"\nAverage Tracking Error: {avg_error:.2f} pixels")

    # Comment based on error
    if avg_error < 5:
        print("🔍 Excellent tracking accuracy. The Kalman filter tracks the pupil very well.")
    elif avg_error < 10:
        print("✅ Good tracking accuracy. Small fluctuations detected, but overall smooth.")
    else:
        print("⚠️ Tracking is unstable. Consider tuning the filter or improving detection.")
else:
    print("No detections were made. Check lighting/camera or adjust HoughCircles params.")
