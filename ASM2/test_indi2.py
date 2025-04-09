import cv2
import numpy as np
from collections import deque
import math

errors_left = deque(maxlen=1000)  # Store last 1000 errors
errors_right = deque(maxlen=1000)


# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Kalman Filter setup
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

# Initialize two Kalman filters for two eyes
kf_left = create_kalman()
kf_right = create_kalman()

def detect_eyes_and_pupils_with_kalman(frame):
    global errors_left, errors_right  # <-- access global error trackers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    pupil_centers = [None, None]
    predictions = []

    for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Up to 2 eyes
        roi = frame[ey:ey+eh, ex:ex+ew]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=50, param2=15, minRadius=2, maxRadius=15)

        kalman = kf_left if i == 0 else kf_right
        errors = errors_left if i == 0 else errors_right

        if circles is not None:
            x, y, r = np.round(circles[0, 0]).astype("int")
            abs_x, abs_y = ex + x, ey + y
            kalman.correct(np.array([[np.float32(abs_x)], [np.float32(abs_y)]]))
            pupil_centers[i] = (abs_x, abs_y)
        else:
            pupil_centers[i] = None

        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        predictions.append((pred_x, pred_y))

        # Draw results
        if pupil_centers[i] is not None:
            cv2.circle(frame, pupil_centers[i], 4, (0, 255, 0), -1)
            error = math.hypot(pupil_centers[i][0] - pred_x, pupil_centers[i][1] - pred_y)
            errors.append(error)
        else:
            cv2.circle(frame, (pred_x, pred_y), 4, (0, 0, 255), -1)

    # Show average error
    if errors_left:
        avg_error_left = sum(errors_left) / len(errors_left)
        cv2.putText(frame, f'Left Error: {avg_error_left:.1f}px', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if errors_right:
        avg_error_right = sum(errors_right) / len(errors_right)
        cv2.putText(frame, f'Right Error: {avg_error_right:.1f}px', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame

# Main loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = detect_eyes_and_pupils_with_kalman(frame)
    cv2.imshow("Kalman Eye Pupil Tracker", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
