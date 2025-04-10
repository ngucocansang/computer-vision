import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def generate_point_cloud_from_stereo(imgL_path, imgR_path, focal_length=880, baseline=0.05):
    imgL_gray = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR_gray = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    imgL_color = cv2.imread(imgL_path)

    if imgL_gray is None or imgR_gray is None:
        raise FileNotFoundError("❌ One or both stereo images not found.")

    stereo = cv2.StereoSGBM_create(
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        minDisparity=50,
        numDisparities=70,
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=1
=======
        minDisparity=10,
        numDisparities=128,  # Must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
=======
        minDisparity=10,
        numDisparities=128,  # Must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
>>>>>>> Stashed changes
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=50,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    )

    disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0
    print("📏 Disparity range:", np.min(disparity), "to", np.max(disparity))

    disparity = cv2.bilateralFilter(disparity, 9, 75, 75)

    plt.figure(figsize=(10, 4))
    plt.imshow(disparity, cmap='inferno')
    plt.colorbar(label='Disparity')
    plt.title("Filtered Disparity Map")
    plt.show()

    h, w = imgL_gray.shape
    Q = np.float32([
        [1, 0, 0, -w / 2],
        [0, -1, 0, h / 2],
        [0, 0, 0, -focal_length],
        [0, 0, 1 / baseline, 0]
    ])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)

    mask = (disparity > 1) & (disparity < np.max(disparity))
    output_points = points_3D[mask]
    output_colors = colors[mask]

    dists = np.linalg.norm(output_points, axis=1)
    output_points = output_points[dists < 4.0]
    output_colors = output_colors[dists < 4.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(output_colors.astype(np.float32) / 255.0)

    return filter_and_clean_point_cloud(pcd)


def generate_point_cloud_from_depth(color_path, depth_path, camera_params, depth_scale=4000.0, depth_trunc=3.0):
    fx, fy, cx, cy, R, T = camera_params
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if color is None or depth is None:
        raise FileNotFoundError("❌ Could not load color or depth image.")

    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) / depth_scale
    mask = (z > 0) & (z < depth_trunc)

    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    x, y, z = x[mask], y[mask], z[mask]

    xyz = np.vstack((x, y, z)).T
    xyz = np.dot(R, xyz.T).T + T

    rgb = color[j[mask], i[mask]]
    rgb = rgb[:, ::-1] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return filter_and_clean_point_cloud(pcd)


if __name__ == "__main__":
    camera_params = (
        880.0, 990.0, 430.0, 540.0,  # fx, fy, cx, cy
        np.eye(3),
        np.zeros(3)
    )

    # === Stereo Reconstruction ===
    try:
        pcd_stereo = generate_point_cloud_from_stereo("left_1.jpg", "right_1.jpg")
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        o3d.io.write_point_cloud("stereo_output.ply", pcd_stereo)
        print("✅ Saved stereo point cloud to 'stereo_output.ply'")
=======
        o3d.io.write_point_cloud("stereo_output_cleaned.ply", pcd_stereo)
        print("✅ Stereo point cloud saved.")
>>>>>>> Stashed changes
=======
        o3d.io.write_point_cloud("stereo_output_cleaned.ply", pcd_stereo)
        print("✅ Stereo point cloud saved.")
>>>>>>> Stashed changes
        o3d.visualization.draw_geometries([pcd_stereo])
    except Exception as e:
        print("Stereo error:", e)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
    # ===== RGB-D Example (Uncomment to use) =====
    camera_params = (
        880.0, 990.0, 430.0, 540.0,  # fx, fy, cx, cy
        np.eye(3),                   # Rotation
        np.zeros(3)                 # Translation
    )
    # try:
    #     # pcd_depth = generate_point_cloud_from_depth("color.png", "depth.png", camera_params)
    #     # //o3d.io.write_point_cloud("depth_output.ply", pcd_depth)
    #     # print("✅ Saved depth point cloud to 'depth_output.ply'")
    #     # o3d.visualization.draw_geometries([pcd_depth])
    # except Exception as e:
    #     print("Depth error:", e)
=======
=======
>>>>>>> Stashed changes
    # === Depth Reconstruction ===
    try:
        pcd_depth = generate_point_cloud_from_depth("color_img.png", "depth_img.png", camera_params)
        o3d.io.write_point_cloud("depth_output_cleaned.ply", pcd_depth)
        print("✅ Depth point cloud saved.")
        o3d.visualization.draw_geometries([pcd_depth])
    except Exception as e:
        print("Depth error:", e)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
