import json
import numpy as np
import cv2
import open3d as o3d

# 1. Load camera parameters từ file JSON
with open("camera_params.json", "r") as f:
    camera_params = json.load(f)

color_intrinsics = camera_params["color_intrinsics"]
depth_intrinsics = camera_params["depth_intrinsics"]
extrinsics = camera_params["extrinsics"]


# 2. Chuyển ma trận ngoại tại (extrinsics) thành 4x4
extrinsics_matrix = np.eye(4)
extrinsics_matrix[:3, :3] = np.array(extrinsics["rotation"]).reshape(3, 3)
extrinsics_matrix[:3, 3] = np.array(extrinsics["translation"])

# 3. Load ảnh màu và ảnh độ sâu
color_img = cv2.imread("color.png", cv2.IMREAD_COLOR)
depth_img = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)

# Kiểm tra nếu ảnh độ sâu có vấn đề (giá trị NaN hoặc 0)
depth_img = np.where(depth_img > 0, depth_img.astype(np.float32) / 1000.0, np.nan)  # mm -> meters

# 4. Trích xuất thông số nội tại của camera
fx, fy = depth_intrinsics["fx"], depth_intrinsics["fy"]
cx, cy = depth_intrinsics["ppx"], depth_intrinsics["ppy"]

# 5. Tạo Point Cloud từ ảnh độ sâu và ảnh màu
def create_point_cloud(color_img, depth_img, fx, fy, cx, cy):
    h, w = depth_img.shape
    points, colors = [], []

    for v in range(h):
        for u in range(w):
            z = depth_img[v, u]
            if np.isnan(z) or z <= 0:  # Loại bỏ điểm không có độ sâu
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append((x, y, z))
            colors.append(color_img[v, u][::-1] / 255.0)  # Chuẩn hóa màu về [0,1]

    points = np.array(points)
    # colors.append = ([r,g,b])
    colors = np.array(colors)

    # 6. Chuyển đổi sang Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def rotate_point_cloud(pcd, angle_deg):
    angle_rad = np.deg2rad(angle_deg)  # Chuyển đổi sang radian
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad), np.cos(angle_rad),  0],
                  [0,                0,                 1]])  # Ma trận xoay quanh trục Z

    # Lấy dữ liệu điểm
    points = np.asarray(pcd.points)
    
    # Áp dụng phép biến đổi
    rotated_points = np.dot(points, R.T)
    
    # Cập nhật đám mây điểm
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    return pcd

# 7. Tạo Point Cloud với ảnh đã căn chỉnh
aligned_pcd = create_point_cloud(color_img, depth_img, fx, fy, cx, cy)
aligned_pcd = rotate_point_cloud(aligned_pcd, 180)
aligned_pcd, _ = aligned_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)


# 8. Biến đổi point cloud theo ma trận ngoại tại
aligned_pcd.transform(extrinsics_matrix)

# 9. Hiển thị kết quả
o3d.visualization.draw_geometries([aligned_pcd])
