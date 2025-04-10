import cv2
import numpy as np

# === Camera Setup ===
cap_left = cv2.VideoCapture(0)   # Adjust camera indexes if needed
cap_right = cv2.VideoCapture(1)

# Set resolution (should be the same for both cameras)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# StereoSGBM parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=20,
    numDisparities=64,
    blockSize=9,
    P1=8 * 3 * 9**2,
    P2=32 * 3 * 9**2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=1
)

# Camera calibration parameters
focal_length = 800   # in pixels
baseline = 0.05      # in meters
threshold_cm = 50    # warning if object is closer than this

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()

    if not retL or not retR:
        print("❌ Failed to capture frames.")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # === Compute Disparity Map ===
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # === Convert Disparity to Depth ===
    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / disparity
    depth_map[np.isinf(depth_map)] = 0
    depth_map[np.isnan(depth_map)] = 0

    # === Warning System ===
    warning_mask = (depth_map < (threshold_cm / 100.0)) & (depth_map > 0)
    num_close = np.count_nonzero(warning_mask)

    if num_close > 0:
        min_dist = np.min(depth_map[depth_map > 0])
        print(f"🚨 Object too close! Closest: {min_dist*100:.2f} cm")

    # === Display Results ===
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    danger_vis = np.uint8(warning_mask * 255)

    cv2.imshow("Left View", frameL)
    cv2.imshow("Disparity", disp_vis)
    cv2.imshow("⚠️ Danger Zones (< 50cm)", danger_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
