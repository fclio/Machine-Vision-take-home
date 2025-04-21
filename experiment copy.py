import json
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2

def visualize_centered_points(points_centered, centroid):
    # Create a point cloud from centered points
    pcd_centered = o3d.geometry.PointCloud()
    pcd_centered.points = o3d.utility.Vector3dVector(points_centered)
    
    # Paint the centered points in a specific color (e.g., blue)
    pcd_centered.paint_uniform_color([0, 0, 1])  # Blue color for centered points
    
    # Create a sphere to visualize the centroid at (0, 0, 0)
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    origin_sphere.translate(np.array([0, 0, 0]))  # Position the sphere at the origin
    origin_sphere.paint_uniform_color([1, 0, 0])  # Red color for the origin
    
    # Visualize the centered points and the origin
    o3d.visualization.draw_geometries([pcd_centered, origin_sphere])

    # Print the centroid of the centered points
    print(f"Centroid of centered points: {centroid}")
    print(f"New centroid position: {np.mean(points_centered, axis=0)}")


 # Function to visualize the point cloud, centroid, PCA axes, and the origin (0,0,0)
 # Function to visualize the PCA axes and the brick axes

def visualize_pca_and_brick_axes(pcd, pca_axes, brick_dimensions, centroid):
    # Create visualization
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Color the point cloud gray
    
    # Visualize the PCA axes as arrows
    arrow_length = 0.1  # Adjust length of PCA axes
    pca_arrows = []
    for i in range(3):
        start = centroid
        end = centroid + pca_axes[i] * arrow_length
        pca_arrows.append(o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([start, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])))
    
    # Set colors for PCA arrows
    pca_arrows[0].paint_uniform_color([1, 0, 0])  # X-axis (Red)
    pca_arrows[1].paint_uniform_color([0, 1, 0])  # Y-axis (Green)
    pca_arrows[2].paint_uniform_color([0, 0, 1])  # Z-axis (Blue)

    # Visualize the brick axes as arrows
    brick_arrow_length = 0.1  # Adjust length of brick axes
    brick_arrows = []
    for i in range(3):
        start = centroid
        end = centroid + np.array([brick_dimensions[i], 0, 0]) if i == 0 else centroid + np.array([0, brick_dimensions[i], 0]) if i == 1 else centroid + np.array([0, 0, brick_dimensions[i]])
        brick_arrows.append(o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([start, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])))

    # Set colors for brick arrows
    brick_arrows[0].paint_uniform_color([1, 0.5, 0])  # X-axis (Orange)
    brick_arrows[1].paint_uniform_color([0, 1, 0.5])  # Y-axis (Light Green)
    brick_arrows[2].paint_uniform_color([0.5, 0, 1])  # Z-axis (Purple)
    
    # Visualize the centroid
    centroid_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid_point.translate(centroid)
    centroid_point.paint_uniform_color([1, 1, 0])  # Yellow color for centroid
    
    # Show the point cloud, PCA axes, brick axes, and centroid
    o3d.visualization.draw_geometries([pcd] + pca_arrows + brick_arrows + [centroid_point])

def visualize_point_cloud(pcd):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)
    o3d.visualization.draw_geometries([pcd, axis, bbox])


def visualize_brick_with_pose_and_pcd(translation, rotation_matrix, pcd, brick_dimensions=(210, 100, 50)):
    # Convert brick dimensions to meters (since translation is in mm)
    brick_dimensions_meters = np.array(brick_dimensions) / 1000.0  # Convert mm to meters
    
    # Create a 3D box to represent the brick
    box = o3d.geometry.TriangleMesh.create_box(brick_dimensions_meters[0], brick_dimensions_meters[1], brick_dimensions_meters[2])

    # Apply rotation to the box
    box.rotate(rotation_matrix, center=box.get_center())
    
    # Apply translation to the box (convert to meters)
    box.translate(np.array(translation) / 1000.0)  # Convert translation from mm to meters
    
    # Create a coordinate frame to represent the camera's frame at (0, 0, 0)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize the brick, the coordinate frame (camera), and the point cloud
    o3d.visualization.draw_geometries([box, coordinate_frame, pcd])

# Function to load camera intrinsic parameters from a JSON file
def load_camera_intrinsics(cam_params):

    fx = cam_params['fx']
    fy = cam_params['fy']
    cx = cam_params['px']
    cy = cam_params['py']
    width = cam_params['width']
    height = cam_params['height']

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    return intrinsic

# Function to load RGB-D images and extract point cloud
import numpy as np
import open3d as o3d

import numpy as np
import open3d as o3d

import numpy as np
import open3d as o3d

import numpy as np
import open3d as o3d

def filter_point_cloud_center(pcd, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(0.3, 1.0)):
    """
    Filter points that are within a central box region in 3D space.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Define filter mask
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )

    # Apply mask
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered_pcd

import numpy as np
import open3d as o3d

import numpy as np
import open3d as o3d

def rotate_point_cloud_90(pcd):
    # Convert point cloud to NumPy array
    points = np.asarray(pcd.points)
    
    # Step 1: Find the center of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Step 2: Translate points to the origin (subtract centroid)
    points_centered = points - centroid
    
    # Step 3: Define the rotation matrix for 90 degrees around the Z-axis
    rotation_matrix = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
    
    # Step 4: Apply the rotation to the centered points
    points_rotated = points_centered @ rotation_matrix.T
    
    # Step 5: Translate the points back to the original position
    points_rotated = points_rotated + centroid
    
    # Update the point cloud with the rotated points
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(points_rotated)
    
    return pcd_rotated



def fit_brick_orientation(pcd, brick_dims=(0.21, 0.1, 0.05)):
 
    # Step 2: Try both orientations (long-x or short-x)
    # def get_fit_error(pcd_points, dims):
    #     bbox = np.max(pcd_points, axis=0) - np.min(pcd_points, axis=0)
    #     return np.linalg.norm(np.array(dims) - bbox)
 
    def get_fit_error(pcd_points, dims):
        points = np.asarray(pcd_points.points)
        # Extract the Y-coordinates of the point cloud
        y_coords = points[:, 1]
        
        # Get the minimum and maximum Y values
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        # Calculate the span along the Y-axis
        y_span = y_max - y_min
        
        # Compare the Y span to the brick width
        error = np.abs(dims[1] - y_span)
        
        return error

    dims1 = brick_dims  # (210, 100, 50)
    dims2 = (brick_dims[1], brick_dims[0], brick_dims[2])  # (100, 210, 50)

    error1 = get_fit_error(pcd, dims1)
    error2 = get_fit_error(pcd, dims2)
    print("error 1", error1)
    print("error 2", error2)
    if error1 < error2:
        best_dims = dims1
        orientation = "long_side_aligned_with_X"
        pcd_rotated = pcd

    else:
        # Rotate around Z by 90° to flip X and Y
        pcd_rotated = rotate_point_cloud_90(pcd)
        pcd_rotated = filter_point_cloud_center(pcd_rotated, x_range=(-0.2, 0.2), y_range=(0.35, 0.45), z_range=(-0.1, 0.1))
        best_dims = dims2
        orientation = "short_side_aligned_with_X"
    
    visualize_point_cloud(pcd_rotated)
    # Step 3: Transform back to world frame
    # pcd_result = pcd_rotated.translate(centroid)
    return pcd_rotated, orientation


def load_point_cloud(depth_image_path, color_image_path, intrinsic):
    # Read the depth and color images
    depth_raw = o3d.io.read_image(depth_image_path)
    color_raw = o3d.io.read_image(color_image_path) if color_image_path else None
    
    depth_scale = 10000.0  # 0.1mm
    
    # Create the RGBD image from the cropped depth and color images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depth_scale, convert_rgb_to_intensity=False)

    # Create point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Flip for Open3D convention for our task (Open3D uses camera facing -Z)
    pcd.transform([[1,  0,  0, 0],
               [0,  0, 1, 0],
               [0,  -1,  0, 0],
               [0,  0,  0, 1]])

    
    return pcd

# Function to estimate pose using PCA
def estimate_pose(pcd, brick_dimensions=(0.21, 0.1, 0.05)):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Center the point cloud (subtract the centroid)
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    visualize_centered_points(points_centered, centroid)
    
    # Perform PCA to get the principal axes
    pca = PCA(n_components=3)
    pca.fit(points_centered)


    # Get the rotation matrix from PCA components
    pca_axes = pca.components_

 
    
    # Align the PCA axes with the known brick dimensions
    # Assuming we know the brick dimensions in X, Y, Z (210x100x50mm)
    # brick_axes = np.array([brick_dimensions, [0, 0, 0]])  # Define brick axes

    brick_axes = np.array([
        [brick_dimensions[0], 0, 0],  # X-axis (brick length)
        [0, brick_dimensions[1], 0],  # Y-axis (brick width)
        [0, 0, brick_dimensions[2]]   # Z-axis (brick height)
    ])
    
    # visualize_pca_and_brick_axes(pcd, pca_axes, brick_dimensions, centroid)

    # Calculate the rotation matrix by aligning the PCA axes with the brick's known axes
    rotation_matrix = np.dot(pca_axes.T, brick_axes.T)
    
    # for i in range(3):
    #     norm = np.linalg.norm(rotation_matrix[:, i])
    #     rotation_matrix[:, i] /= norm  # Normalize the column

    # Adjust the rotation to match the new coordinate system
    # Swap Y and Z axes in the rotation matrix to match the coordinate system
    # still need to check with other images!!!
    # rotation_matrix = np.array([
    #     [rotation_matrix[0, 0], rotation_matrix[0, 2], rotation_matrix[0, 1]],  # Swap Y and Z for X row
    #     [rotation_matrix[1, 0], rotation_matrix[1, 2], rotation_matrix[1, 1]],  # Swap Y and Z for Y row
    #     [rotation_matrix[2, 0], rotation_matrix[2, 2], rotation_matrix[2, 1]]   # Swap Y and Z for Z row
    # ])

    # Translation: the centroid gives the brick's position relative to the camera
    
    translation = centroid * 1000 # The translation is simply the center position
    

    # Convert rotation matrix to Euler angles (Roll, Pitch, Yaw)
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    print("angle", roll, pitch, yaw)

    # rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    # r_2 = R.from_matrix(rot_matrix)
    # roll_2, pitch_2, yaw_2 = r_2.as_euler('xyz', degrees=True)
    # print("angle 2", roll_2, pitch_2, yaw_2)


    # a = np.allclose(rotation_matrix, rot_matrix, atol=1e-6)

    # print("rot matrix", rot_matrix)
    print("rotation matrix", rotation_matrix)
    
    # visualize_brick_with_pose_and_pcd(translation, rot_matrix, pcd)
    visualize_brick_with_pose_and_pcd(translation, rotation_matrix, pcd)
    
    
    # translation = [translation[0], -translation[2], translation[1]]
    

    
    return roll, pitch, yaw, translation, rotation_matrix, pca_axes, centroid, points_centered


# Example usage:
# Assuming pcd, pca_axes, and centroid are already computed
# visualize_pose_with_origin(pcd, pca_axes, centroid)

import open3d as o3d


def compute_rotation_translation(depth_image_path, color_image_path, cam_params):

    intrinsic= load_camera_intrinsics(cam_params)
    # Load point cloud
    pcd = load_point_cloud(depth_image_path, color_image_path, intrinsic)
    
    # visualize_point_cloud(pcd)

    # what if they place the box far away or close up. 
    pcd = filter_point_cloud_center(pcd, x_range=(-0.15, 0.15), y_range=(0.2, 0.55), z_range=(-0.1, 0.1))

    visualize_point_cloud(pcd)

    # pcd, orientation = fit_brick_orientation(pcd)
    # print("Detected brick orientation:", orientation)

    
    # visualize_point_cloud(pcd)
    
    # # Estimate pose and get PCA axes, centroid, and centered points
    roll, pitch, yaw, translation, rot_matrix, pca_axes, centroid, points_centered = estimate_pose(pcd)

    # if orientation == "short_side_aligned_with_X":
    #     yaw += 90

    # Print estimated pose
    print(f"Estimated Pose: Roll = {roll}, Pitch = {pitch}, Yaw = {yaw}")
    print(f"Translation: {translation}")

    # Visualize the result
    return translation, [roll, pitch, yaw],rot_matrix
