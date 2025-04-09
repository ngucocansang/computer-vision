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
dt = 1  # time step
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0 ],
              [0, 0, 0, 1 ]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Initial state and matrices
x = np.zeros((4, 1))  # initial state: x, y, vx, vy
P = np.eye(4) * 1000  # initial uncertainty
Q = np.eye(4) * 0.1   # process noise
R = np.array([[12515.22, 0], [0, 32843.24]])  # from task (a), corrected

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    measurement = detect_pupil(frame)

    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    if measurement is not None:
        z = np.array([[measurement[0]], [measurement[1]]])

        # Kalman Gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Update
        y = z - H @ x
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        # Draw green circle (actual detection)
        cv2.circle(frame, (int(z[0]), int(z[1])), 20, (0, 255, 0), 2)

        # Draw blue line between prediction and detection
        cv2.line(frame, (int(x[0]), int(x[1])), (int(z[0]), int(z[1])), (255, 255, 0), 2)

    # Draw red dot (Kalman prediction)
    predicted_pos = (int(x[0]), int(x[1]))
    cv2.circle(frame, predicted_pos, 5, (0, 0, 255), -1)

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
