import cv2
import numpy as np

# --- Stereo Camera Calibration Parameters ---
FOCAL_LENGTH = 6745.82       # pixels
BASELINE = 10.0              # cm
DISTANCE_THRESHOLD = 40.0    # cm

# --- Disparity Parameters ---
min_disp = 0
num_disp = 128  # must be divisible by 16
block_size = 7

def create_stereo_matcher():
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

def detect_red_object(hsv_frame):
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

def compute_distance(points_3D, mask, max_distance=1000):
    points_roi = points_3D[mask == 255]
    z_vals = points_roi[:, 2]
    valid_z = z_vals[(z_vals > 0) & (z_vals < max_distance)]
    return np.median(valid_z) if len(valid_z) > 0 else None

def main():
    # --- Load calibration data ---
    calib = np.load("stereo_calib_data.npz")
    Q = calib["Q"]
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        calib["M1"], calib["D1"], calib["R1"], calib["P1"], (640, 480), cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        calib["M2"], calib["D2"], calib["R2"], calib["P2"], (640, 480), cv2.CV_16SC2)

    # --- Setup stereo matcher ---
    stereo = create_stereo_matcher()

    # --- Open cameras ---
    left_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    right_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    if not left_cam.isOpened() or not right_cam.isOpened():
        print("Error: Could not open camera(s)")
        return

    while True:
        ret_left, frame_left = left_cam.read()
        ret_right, frame_right = right_cam.read()
        if not ret_left or not ret_right:
            print("Frame capture error")
            break

        frame_left = cv2.resize(frame_left, (640, 480))
        frame_right = cv2.resize(frame_right, (640, 480))

        hsv_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
        red_mask = detect_red_object(hsv_left)

        # Rectify grayscale images
        gray_left = cv2.remap(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY),
                              left_map1, left_map2, cv2.INTER_LINEAR)
        gray_right = cv2.remap(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY),
                               right_map1, right_map2, cv2.INTER_LINEAR)

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # Normalized disparity for visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        result = cv2.bitwise_and(frame_left, frame_left, mask=red_mask)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            roi_mask = np.zeros_like(red_mask)
            cv2.drawContours(roi_mask, [largest], -1, 255, -1)

            distance = compute_distance(points_3D, roi_mask)

            distance_text = f"Distance: {distance:.2f} cm" if distance else "Distance: N/A"
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)

            if distance and distance < DISTANCE_THRESHOLD:
                cv2.putText(result, "WARNING: OBJECT TOO CLOSE!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # --- Display ---
        cv2.imshow("Red Object Detection", result)
        cv2.imshow("Right Camera", frame_right)
        cv2.imshow("Disparity Map", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
