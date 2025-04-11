import cv2
import numpy as np

# --- Stereo Camera Calibration Parameters ---
FOCAL_LENGTH = 480       # in pixels (based on your setup)
BASELINE = 10           # in cm
DISTANCE_THRESHOLD = 40.0  # in cm

# --- Depth from Disparity using MEDIAN for robustness ---
def depth_from_disparity(disparity_values, focal_length, baseline):
    valid_disparities = disparity_values[(disparity_values > 0) & (disparity_values < 128)]  # clamp upper bound
    if len(valid_disparities) == 0:
        return None
    median_disp = np.median(valid_disparities)
    return (focal_length * baseline) / median_disp

# --- Use StereoSGBM for better disparity quality ---
min_disp = 90
num_disp = 128  # must be divisible by 16
block_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# --- Open Cameras ---
left_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
right_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while True:
    ret_left, frame_left = left_cam.read()
    ret_right, frame_right = right_cam.read()

    if not ret_left or not ret_right:
        print("Camera error")
        break

    frame_left = cv2.resize(frame_left, (640, 480))
    frame_right = cv2.resize(frame_right, (640, 480))

    # --- Blue Object Detection ---
    hsv = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(frame_left, frame_left, mask=red_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- Disparity Map (Raw, Unnormalized) ---
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    raw_disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # scale back

    # --- Optional: Normalized disparity for display ---
    disp_display = cv2.normalize(raw_disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_display = np.uint8(disp_display)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # --- Mask disparity map inside object region ---
        roi_mask = np.zeros_like(red_mask)
        cv2.drawContours(roi_mask, [largest_contour], -1, 255, -1)
        disparity_roi = raw_disparity[roi_mask == 255]

        # --- Improved Distance Estimation ---
        distance = depth_from_disparity(disparity_roi, FOCAL_LENGTH, BASELINE)

        if distance is not None:
            distance_text = f"Distance: {distance:.2f} cm"
        else:
            distance_text = "Distance: N/A"

        # --- Draw overlays ---
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        if distance is not None and distance < DISTANCE_THRESHOLD:
            cv2.putText(result, "WARNING: OBJECT TOO CLOSE!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # --- Display Windows ---
    cv2.imshow("Blue Detection (Left)", result)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disp_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()