import open3d as o3d
import numpy as np
import cv2
import sys

# === CONFIG ===
COLOR_PATH = "left_group.jpg"  # Replace with your RGB image path
DEPTH_PATH = "right_group.png"  # Replace with your depth image path (16-bit PNG)
OUTPUT_PCD = "output.ply" # Output point cloud filename
DEPTH_SCALE = 1000.0      # 1 meter = 1000 mm (RealSense, Kinect, etc.)
FX = 615.0                # Adjust based on your camera
FY = 615.0
CX = 320.0
CY = 240.0

# === LOAD IMAGES ===
color_raw = cv2.imread(COLOR_PATH)
depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

if color_raw is None or depth_raw is None:
    sys.exit("❌ Error loading images. Check file paths.")

# Resize for alignment (optional, only if needed)
if color_raw.shape[:2] != depth_raw.shape[:2]:
    depth_raw = cv2.resize(depth_raw, (color_raw.shape[1], color_raw.shape[0]))

# Convert BGR to RGB
color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)

# === CREATE RGBD IMAGE ===
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color_raw),
    o3d.geometry.Image(depth_raw),
    depth_scale=DEPTH_SCALE,
    convert_rgb_to_intensity=False
)

# === INTRINSIC MATRIX ===
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=color_raw.shape[1],
    height=color_raw.shape[0],
    fx=FX, fy=FY,
    cx=CX, cy=CY
)

# === GENERATE POINT CLOUD ===
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic
)

# Flip it to match OpenCV coordinate system
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# === FILTERING & DOWNSAMPLING (optional) ===
pcd = pcd.voxel_down_sample(voxel_size=0.0025)  # Smaller = higher detail
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)

# === SAVE & VISUALIZE ===
o3d.io.write_point_cloud(OUTPUT_PCD, pcd, write_ascii=True)
o3d.visualization.draw_geometries([pcd], window_name="3D Output", width=800, height=600)
