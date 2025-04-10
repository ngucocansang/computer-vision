import cv2
import numpy as np

# CONFIG
DEPTH_SCALE = 4000.0  # e.g., RealSense gives depth in 1/4000 m
WARNING_DISTANCE = 0.5  # meters

# Open camera (use ID 0 for normal webcam or special ID for depth camera)
cap_color = cv2.VideoCapture(0)  # RGB camera
cap_depth = cv2.VideoCapture(1)  # Depth stream (make sure your device supports this)

if not cap_color.isOpened() or not cap_depth.isOpened():
    raise RuntimeError("❌ Could not access one or both video streams.")

while True:
    ret_color, color_frame = cap_color.read()
    ret_depth, depth_frame = cap_depth.read()

    if not ret_color or not ret_depth:
        print("⚠️ Frame capture failed.")
        break

    # Convert depth to meters
    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    depth_meters = depth_gray.astype(np.float32) / DEPTH_SCALE

    # Mask invalid depth values
    valid_mask = (depth_meters > 0) & (depth_meters < 5.0)
    if np.count_nonzero(valid_mask) == 0:
        min_distance = None
    else:
        min_distance = np.min(depth_meters[valid_mask])

    # Draw distance info
    if min_distance is not None:
        if min_distance < WARNING_DISTANCE:
            cv2.putText(color_frame, f'⚠ TOO CLOSE! ({min_distance*100:.0f} cm)',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(color_frame, f'Distance: {min_distance:.2f} m',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(color_frame, "No depth data", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Visualize depth
    depth_visual = cv2.applyColorMap(cv2.convertScaleAbs(depth_meters * 255 / 5.0, alpha=1), cv2.COLORMAP_JET)

    cv2.imshow("Live Color View", color_frame)
    cv2.imshow("Live Depth View", depth_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_color.release()
cap_depth.release()
cv2.destroyAllWindows()
