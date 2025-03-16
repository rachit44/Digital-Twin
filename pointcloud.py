import numpy as np
import open3d as o3d
import cv2
def filter_point_cloud(pcd, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    # Voxel Downsampling
    down_pcd = pcd.voxel_down_sample(voxel_size)

    # Statistical Outlier Removal
    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    filtered_pcd = down_pcd.select_by_index(ind)
    return filtered_pcd


def generate_point_cloud(rgb_image, depth_map, scale=1.0):
    """
    Generates an Open3D point cloud from an RGB image and depth map.

    Args:
        rgb_image (np.ndarray): Original BGR image (from OpenCV).
        depth_map (np.ndarray): Normalized depth map [0-1].
        scale (float): Scaling factor for depth values.

    Returns:
        o3d.geometry.PointCloud: Generated point cloud.
    """
    # Convert BGR to RGB for Open3D
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    height, width = depth_map.shape
    fx = fy = max(width, height) * 0.8  # Adjust focal length
    cx = width / 2
    cy = height / 2

    # Create a meshgrid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten to vectors
    u = u.flatten()
    v = v.flatten()
    z = depth_map.flatten() * scale

    # Filter out near-zero depth values (optional, improves clarity)
    valid = z > 0.01
    z = z[valid]
    u = u[valid]
    v = v[valid]

    # Compute x, y, z points in camera space
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy

    # Stack points (N, 3)
    points = np.vstack((x, y, z)).transpose()

    # Get colors
    colors = rgb_image[v, u] / 255.0  # Normalize RGB

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


