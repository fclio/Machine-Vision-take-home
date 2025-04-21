import cv2
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from visualization import *

def load_camera_intrinsics(cam_params):
    """Create Open3D intrinsic matrix from camera parameters."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=cam_params['width'], height=cam_params['height'],
        fx=cam_params['fx'], fy=cam_params['fy'],
        cx=cam_params['px'], cy=cam_params['py']
    )
    return intrinsic


def filter_point_cloud_center(pcd, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(0.3, 1.0)):
    """Crop a region of interest from the point cloud."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Apply mask to filter points within the specified ranges
    mask = (
        (x_range[0] <= points[:, 0]) & (points[:, 0] <= x_range[1]) &
        (y_range[0] <= points[:, 1]) & (points[:, 1] <= y_range[1]) &
        (z_range[0] <= points[:, 2]) & (points[:, 2] <= z_range[1])
    )

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[mask])
    filtered.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered


def load_point_cloud(depth_image_data, color_image_data, intrinsic):
    """Generate point cloud from depth and RGB images using Open3D."""
    if color_image_data is not None:
        color_image_data = cv2.cvtColor(color_image_data, cv2.COLOR_BGR2RGB)

    depth_raw = o3d.geometry.Image(depth_image_data.astype(np.uint16))
    color_raw = o3d.geometry.Image(color_image_data.astype(np.uint8)) if color_image_data is not None else None

    depth_scale = 10000.0  # Convert 0.1mm depth units to meters

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Flip Y and Z to align with Open3D's visual convention
    pcd.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return pcd


def estimate_pose(pcd, visualize=False, brick_dimensions=(0.21, 0.1, 0.05)):
    """Estimate the pose (rotation and translation) of the brick from the point cloud."""
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    if visualize:
        visualize_centered_points(points_centered, centroid)

    # PCA to estimate main axes
    pca = PCA(n_components=3)
    pca.fit(points_centered)
    pca_axes = pca.components_

    if visualize:
        visualize_pca_and_brick_axes(pcd, pca_axes, brick_dimensions, centroid)

    # Estimate rotation matrix and translation
    brick_axes = np.array([[brick_dimensions[0], 0, 0], [0, brick_dimensions[1], 0], [0, 0, brick_dimensions[2]]])
    rotation_matrix = np.dot(pca_axes.T, brick_axes.T)
    translation = centroid * 1000  # Convert to mm

    # Convert rotation matrix to Euler angles
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)

    if visualize:
        visualize_brick_with_pose_and_pcd(translation, roll, pitch, yaw, pcd)

    return roll, pitch, yaw, translation


def compute_rotation_translation(depth_image, color_image, cam_params, visualize=False):
    """Pipeline for estimating the 6D pose of a brick from RGB-D images."""
    # Load camera intrinsics
    intrinsic = load_camera_intrinsics(cam_params)

    # Load RGB-D data and create point cloud
    pcd = load_point_cloud(depth_image, color_image, intrinsic)

    if visualize:
        visualize_point_cloud(pcd)

    # Filter central region for brick
    pcd = filter_point_cloud_center(pcd, x_range=(-0.15, 0.15), y_range=(0.2, 0.55), z_range=(-0.1, 0.1))
    
    if visualize:
        visualize_point_cloud(pcd)

    # Estimate pose from point cloud
    roll, pitch, yaw, translation = estimate_pose(pcd, visualize=visualize)

    print(f"Estimated Pose: Roll = {roll}, Pitch = {pitch}, Yaw = {yaw}")
    print(f"Translation: {translation}")
    return translation.tolist(), [roll, pitch, yaw]
