import cv2
import numpy as np
import time

def detect_pupil(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and center on the original frame
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            return (x, y)
    return None

cap = cv2.VideoCapture(0)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    center = detect_pupil(frame)

    # Calculate frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display frame rate
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Pupil Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
