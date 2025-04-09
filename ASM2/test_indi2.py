import cv2
import numpy as np
import time

def detect_pupil(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 20,
                                param1=50, param2=30, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0][:2]  # return (x, y)
    return None

# Kalman Filter setup
dt = 1
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0 ],
              [0, 0, 0, 1 ]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

x = np.zeros((4, 1))
P = np.eye(4) * 1000
Q = np.eye(4) * 0.1
R = np.array([[10, 0], [0, 10]])  # Assumed measurement noise covariance

cap = cv2.VideoCapture(0)
prev_time = time.time()
errors = []  # Store tracking errors

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

        # Draw measurement
        cv2.circle(frame, (int(z[0]), int(z[1])), 6, (0, 255, 0), 2)  # Green: measured

        # Calculate error
        error = np.linalg.norm([x[0, 0] - z[0, 0], x[1, 0] - z[1, 0]])
        errors.append(error)

    # Draw prediction
    predicted_pos = (int(x[0]), int(x[1]))
    cv2.circle(frame, predicted_pos, 6, (0, 0, 255), 2)  # Red: predicted

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Kalman Filter - Pupil Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Accuracy results
if errors:
    avg_error = sum(errors) / len(errors)
    print(f"Average Tracking Error: {avg_error:.2f} pixels")

    if avg_error < 5:
        print("🔍 Excellent tracking accuracy. The Kalman filter tracks the pupil very well.")
    elif avg_error < 10:
        print("👍 Good tracking accuracy. The filter handles noise effectively.")
    else:
        print("⚠️ Acceptable tracking, but further tuning might be needed (e.g., better detection or noise adjustment).")
else:
    print("No measurements were detected. Tracking accuracy could not be estimated.")
