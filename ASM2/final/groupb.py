import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def generate_point_cloud_from_stereo(imgL_path, imgR_path, focal_length=800, baseline=0.05):
    imgL_gray = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR_gray = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    imgL_color = cv2.imread(imgL_path, cv2.IMREAD_COLOR)

    if imgL_gray is None or imgR_gray is None:
        raise FileNotFoundError("❌ One or both stereo images not found.")

    stereo = cv2.StereoSGBM_create(
        minDisparity=20,
        numDisparities=50,
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=1
    )

    disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0
    print("📏 Disparity range:", np.min(disparity), "to", np.max(disparity))

   
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap='plasma')
    plt.colorbar(label='Disparity Value')
    plt.title("Disparity Map")
    plt.show()

    h, w = imgL_gray.shape[:2]
    Q = np.float32([
        [1, 0, 0, -w / 2],
        [0, -1, 0, h / 2],
        [0, 0, 0, -focal_length],
        [0, 0, 1 / baseline, 0]
    ])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)

    mask = (disparity > disparity.min()) & (disparity < np.max(disparity))
    output_points = points_3D[mask]
    output_colors = colors[mask]

    dists = np.linalg.norm(output_points, axis=1)
    close_mask = dists < 3.5
    output_points = output_points[close_mask]
    output_colors = output_colors[close_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(output_colors.astype(np.float32) / 255.0)

    return pcd



def generate_point_cloud_from_depth(color_path, depth_path, camera_params, depth_scale=4000.0):
    fx, fy, cx, cy, R, T = camera_params
    color_img = cv2.imread(color_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if color_img is None or depth_img is None:
        raise FileNotFoundError("❌ Error loading depth or color image.")

    h, w = depth_img.shape
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth_img[v, u] / depth_scale
            if z <= 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point = np.array([x, y, z])
            transformed = R @ point + T
            points.append(transformed)
            colors.append(color_img[v, u][::-1] / 255.0)

    if not points:
        raise ValueError("❌ No valid points found in depth data.")

    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# === Main Execution ===
if __name__ == "__main__":
    # ===== Stereo Example =====
    try:
        pcd_stereo = generate_point_cloud_from_stereo("left_1.jpg", "right_1.jpg")
        o3d.io.write_point_cloud("stereo_output.ply", pcd_stereo)
        print("✅ Saved stereo point cloud to 'stereo_output.ply'")
        o3d.visualization.draw_geometries([pcd_stereo])
    except Exception as e:
        print("Stereo error:", e)

    # ===== RGB-D Example (Uncomment and edit paths to use) =====
    camera_params = (
        880.0, 990.0, 430.0, 540.0,  # fx, fy, cx, cy
        np.eye(3),                   # Rotation
        np.zeros(3)                 # Translation
    )
    try:
        pcd_depth = generate_point_cloud_from_depth("left_group.jpg", "right_group.jpg", camera_params)
        o3d.io.write_point_cloud("depth_output.ply", pcd_depth)
        print("✅ Saved RGB-D point cloud to 'depth_output.ply'")
        o3d.visualization.draw_geometries([pcd_depth])
    except Exception as e:
        print("RGB-D error:", e)

    # ===== Stereo Example =====
    try:
        pcd_stereo = generate_point_cloud_from_stereo("left_1.jpg", "right_1.jpg")
        o3d.io.write_point_cloud("stereo_output.ply", pcd_stereo)
        print("✅ Saved stereo point cloud to 'stereo_output.ply'")
        o3d.visualization.draw_geometries([pcd_stereo])
    except Exception as e:
        print("Stereo error:", e)

    # ===== RGB-D Example (Uncomment to use) =====
    camera_params = (
        880.0, 990.0, 430.0, 540.0,  # fx, fy, cx, cy
        np.eye(3),                   # Rotation
        np.zeros(3)                 # Translation
    )
