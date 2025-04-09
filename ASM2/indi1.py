import cv2
import numpy as np
import time
import csv

def detect_pupil(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                param1=50, param2=30, minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        return (x, y)
    return None

# Setup
cap = cv2.VideoCapture(0)
data = []

print("[INFO] Press 'q' to stop and save data.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center = detect_pupil(frame)

    if center:
        x, y = center
        data.append((x, y))
        # Optional: Display coordinates on screen
        cv2.putText(frame, f'({x}, {y})', (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Pupil Detection - Recording Data', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
with open('pupil_positions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Saved {len(data)} detected positions to 'pupil_positions.csv'")
