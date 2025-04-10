import cv2
import numpy as np

# === Calibration Constants ===
KNOWN_DISTANCE = 50.0  # cm
KNOWN_WIDTH = 10.0     # cm
FOCAL_LENGTH = 600     # pixels (calibrated manually)

DISTANCE_THRESHOLD = 20.0  # cm

def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# === Initialize Stereo Matcher ===
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# === Open two cameras ===
left_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
right_cam = cv2.VideoCapture(3, cv2.CAP_DSHOW)

# === Main Loop ===
while True:
    ret_left, frame_left = left_cam.read()
    ret_right, frame_right = right_cam.read()

    if not ret_left or not ret_right:
        print("Camera error")
        break

    # Resize for consistent dimensions (optional)
    frame_left = cv2.resize(frame_left, (640, 480))
    frame_right = cv2.resize(frame_right, (640, 480))

    # ---- Blue Object Detection on Left Camera ----
    hsv = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame_left, frame_left, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Estimate distance using width in pixels
        if w > 0:
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)
            distance_text = f"Distance: {distance:.2f} cm"
        else:
            distance = None
            distance_text = "Distance: N/A"

        # Draw indicators
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.putText(result, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        # Warning if too close
        if distance is not None and distance < DISTANCE_THRESHOLD:
            cv2.putText(result, "WARNING: OBJECT TOO CLOSE!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # ---- Disparity Map from Left and Right Camera ----
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right)
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)

    # ---- Display All Frames ----
    cv2.imshow("Left Camera - Blue Detection", result)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disparity)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
