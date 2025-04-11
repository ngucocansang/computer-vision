import cv2
import numpy as np

# --- Stereo Camera Calibration Parameters ---
FOCAL_LENGTH =  6745.82       # in pixels (based on your setup)
BASELINE = 6.0           # in cm
DISTANCE_THRESHOLD = 50.0  # in cm

# --- Depth Calculation ---
def depth_from_disparity(disparity_values, focal_length, baseline):
    valid_disparities = disparity_values[disparity_values > 0]
    if len(valid_disparities) == 0:
        return None
    mean_disp = np.mean(valid_disparities)
    return (focal_length * baseline) / mean_disp

# --- StereoSGBM Matcher with improved parameters ---
stereo = cv2.StereoSGBM_create(
    minDisparity=20,
    numDisparities=64,  # must be divisible by 16, changed from 50 to 64
    blockSize=9,
    P1=8 * 3 * 9 ** 2,
    P2=32 * 3 * 9 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=1
)

# --- Open Cameras ---
left_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
right_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not left_cam.isOpened() or not right_cam.isOpened():
    print("Cannot open one or both cameras.")
    exit()

while True:
    ret_left, frame_left = left_cam.read()
    ret_right, frame_right = right_cam.read()

    if not ret_left or not ret_right:
        print("Camera error")
        break

    # Resize frames
    frame_left = cv2.resize(frame_left, (640, 480))
    frame_right = cv2.resize(frame_right, (640, 480))

    # --- Blue Object Detection ---
    hsv = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame_left, frame_left, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- Disparity Map ---
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # Proper scaling
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_vis = np.uint8(disparity_vis)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi_mask = np.zeros_like(disparity, dtype=np.uint8)
        cv2.drawContours(roi_mask, [largest_contour], -1, 255, -1)

        # Mask disparity values inside the object
        object_disparity_values = disparity[roi_mask == 255]
        distance = depth_from_disparity(object_disparity_values, FOCAL_LENGTH, BASELINE)

        if distance is not None:
            distance_text = f"Distance: {distance:.2f} cm"
        else:
            distance_text = "Distance: N/A"

        # Draw bounding box and distance
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        # Warning if object too close
        if distance is not None and distance < DISTANCE_THRESHOLD:
            cv2.putText(result, "WARNING: OBJECT TOO CLOSE!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # --- Display ---
    cv2.imshow("Blue Detection", result)
    cv2.imshow("Disparity Map", disparity_vis)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()