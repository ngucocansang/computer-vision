import cv2
import numpy as np

# --- Stereo Camera Calibration Parameters ---
FOCAL_LENGTH = 674.582  # pixels
BASELINE = 10.0         # cm
DISTANCE_THRESHOLD = 50.0  # cm

# --- StereoSGBM Setup ---
stereo = cv2.StereoSGBM_create(
    minDisparity=20,
    numDisparities=128,  # divisible by 16
    blockSize=7,
    P1=8 * 3 * 9 ** 2,
    P2=32 * 3 * 9 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=1
)

def compute_depth(disparity_values, focal_length, baseline):
    valid_disp = disparity_values[(disparity_values > 0) & (disparity_values < 256)]
    if len(valid_disp) < 10:
        return None
    median_disp = np.median(valid_disp)
    if median_disp < 0.1:
        return None  # tránh chia 0
    depth_cm = (focal_length * baseline) / median_disp
    return depth_cm


def detect_blue_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

# --- Open Stereo Cameras ---
left_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
right_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Optional: set resolution instead of resizing later
for cam in [left_cam, right_cam]:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not left_cam.isOpened() or not right_cam.isOpened():
    print("Cannot open cameras.")
    exit()

while True:
    ret_l, frame_left = left_cam.read()
    ret_r, frame_right = right_cam.read()

    if not ret_l or not ret_r:
        print("Camera error.")
        break

    # Disparity Map
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    print(f"Disparity range: min={np.min(disparity)}, max={np.max(disparity)}")

    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_vis = np.uint8(disparity_vis)

    # Detect Blue Object
    mask, contours = detect_blue_object(frame_left)
    result = cv2.bitwise_and(frame_left, frame_left, mask=mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:  # ignore noise
            x, y, w, h = cv2.boundingRect(largest)
            roi_mask = np.zeros(disparity.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, [largest], -1, 255, -1)

            disp_values = disparity[roi_mask == 255]
            distance = compute_depth(disp_values, FOCAL_LENGTH, BASELINE)

            # Display Info
            distance_text = f"Distance: {distance:.2f} cm" if distance else "Distance: N/A"
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
            if distance and distance < DISTANCE_THRESHOLD:
                cv2.putText(result, "WARNING: OBJECT TOO CLOSE!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Show Windows
    cv2.imshow("Blue Detection", result)
    cv2.imshow("Disparity", disparity_vis)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
