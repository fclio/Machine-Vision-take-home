import os
import requests
import json
import numpy as np

import cv2
from scipy.spatial.transform import Rotation as R
import pyvista as pv

import numpy as np


def send_to_server(color_path, depth_path, cam_json_path, endpoint="http://localhost:8000/pose"):
    with open(color_path, "rb") as c, open(depth_path, "rb") as d, open(cam_json_path, "rb") as j:
        files = {
            "color": ("color.png", c, "image/png"),
            "depth": ("depth.png", d, "image/png"),
            "camera": ("cam.json", j, "application/json"),
        }
        response = requests.post(endpoint, files=files)
        return response.json()


# def visualize_pose(depth_img, intrinsics, translation, rotation):
#     import open3d as o3d
#     # Convert depth image to 3D point cloud
#     h, w = depth_img.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_img / 10000.0  # Convert from 0.1 mm to meters
#     x = (i - intrinsics['px']) * z / intrinsics['fx']
#     y = (j - intrinsics['py']) * z / intrinsics['fy']
#     xyz = np.stack((x, y, z), axis=-1)
#     mask = z > 0
#     pts = xyz[mask]

#     # Visualize the point cloud using Open3D
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(pts)

#     # Visualize the translation and rotation (brick pose)
#     rotation_matrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
#     center = np.mean(pts, axis=0)

#     # Create a coordinate frame to visualize rotation
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=center)
#     frame.rotate(rotation_matrix, center=center)

#     # Visualize the point cloud and coordinate frame together
#     o3d.visualization.draw_geometries([point_cloud, frame])
# def visualize_pose_with_camera(depth_img, intrinsics, translation, rotation):
    #   pts = depth_to_point_cloud(depth_img, intrinsics)  
#     # Create the point cloud for the brick
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(pts)

#     # Visualize the translation and rotation (brick pose)
#     rotation_matrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
#     center = np.mean(pts, axis=0)

#     # Create a coordinate frame to visualize the camera pose
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=translation)
#     frame.rotate(rotation_matrix, center=translation)

#     # Visualize the point cloud and the camera pose together
#     o3d.visualization.draw_geometries([point_cloud, frame])


# def visualize_pose(depth_img, intrinsics, translation, rotation):
    # Convert depth image to 3D point cloud
    # h, w = depth_img.shape
    # i, j = np.meshgrid(np.arange(w), np.arange(h))
    # z = depth_img / 10000.0  # Convert from 0.1 mm to meters
    # x = (i - intrinsics['px']) * z / intrinsics['fx']
    # y = (j - intrinsics['py']) * z / intrinsics['fy']
    # xyz = np.stack((x, y, z), axis=-1)
    # mask = z > 0
    # pts = xyz[mask]

    # # Visualize the point cloud using PyVista
    # point_cloud = pv.PolyData(pts)

    # # Create a coordinate frame for translation and rotation visualization
    # rotation_matrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    
    # # Apply translation to the point cloud
    # pts_translated = pts + translation

    # # Create a PyVista PolyData object for the translated point cloud
    # translated_point_cloud = pv.PolyData(pts_translated)

    # # Visualize the translated point cloud
    # plotter = pv.Plotter()
    # plotter.add_mesh(translated_point_cloud, color="blue", point_size=5, render_points_as_spheres=True)

    # # Create and add the coordinate frame (axes) to show rotation
    # origin = np.mean(pts_translated, axis=0)
    # frame = pv.Cube(bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    # frame.points += origin  # Translate the frame to the origin

    # # Rotate the frame based on the rotation matrix
    # frame.rotate_x(rotation[0])
    # frame.rotate_y(rotation[1])
    # frame.rotate_z(rotation[2])

    # plotter.add_mesh(frame, color="red", opacity=0.5)

    # # Set plotter settings
    # plotter.set_background("white")
    # plotter.show()

def visualize_pose(depth_img, intrinsics, translation, rotation):
    """
    Visualize the camera and a brick in 3D space, with the brick's position and rotation applied.
    The brick is placed based on the given translation and rotation.
    """
 
    plotter = pv.Plotter()
    # Add a dot at the origin
    origin_dot = pv.Sphere(radius=5, center=(0, 0, 0))  # Small red dot at origin
    plotter.add_mesh(origin_dot, color="red")

    # Visualize the camera as a bigger unit box (representing the camera)

    camera_width = intrinsics["width"] # Camera box size (in meters, 1m to make it easily visible)
    camera_height = intrinsics["height"]
    camera_box = pv.Cube(bounds=[
                                 -camera_width / 2, camera_width / 2,
                                 -1, 1,
                                 -camera_height / 2, camera_height / 2])
    # camera_box.translate(translation)  # Translate camera to its position
    print(camera_box.points)
    plotter.add_mesh(camera_box, color="green")

    # Visualize the brick with a scaling factor (5x real-world size for better visibility)
    brick_length = 210  
    brick_width = 100   
    brick_height = 50 

    # Create a brick shape (rectangular box)
    brick = pv.Cube(bounds=[-brick_length / 2, brick_length / 2,
                            -brick_width / 2, brick_width / 2,
                            -brick_height / 2, brick_height / 2])
    
    # Apply translation and rotation to place the brick
    # translation[1] += (camera_width / 2)
    # plotter.add_mesh(brick, color="blue", opacity=0.7)
    rot_brick = brick.translate(translation)
    # brick.points += translation # Translate the brick
    # rot_brick = rot_brick.rotate_x(rotation[0])
    # rot_brick = rot_brick.rotate_y(rotation[1])
    # rot_brick = rot_brick.rotate_z(rotation[2])

    # # Add the brick to the plot
    plotter.add_mesh(rot_brick, color="orange", opacity=0.7)

    # Set the background color to white
    plotter.set_background("white")

    # plotter.add_axes_at_origin(line_width=10)
    plotter.add_axes(line_width=5)
    # Show the plot
    plotter.show()

def main():
    dataset_dir = "place_quality_inputs"  # Path to folders with color.png, depth.png, cam.json
    for folder in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(path):
            continue

        print(f"\n📂 Processing {folder}")
        try:
            # Send data to server and get pose estimate
            result = send_to_server(
                os.path.join(path, "color.png"),
                os.path.join(path, "depth.png"),
                os.path.join(path, "cam.json")
            )
            print("✅ Pose:", json.dumps(result, indent=2))

            # Read images and camera intrinsics
            depth_img = cv2.imread(os.path.join(path, "depth.png"), cv2.IMREAD_UNCHANGED)
            with open(os.path.join(path, "cam.json")) as f:
                intrinsics = json.load(f)

            # Visualize the pose and point cloud
            visualize_pose(depth_img, intrinsics, result["translation"], result["rotation"])
            
        except Exception as e:
            print("❌ Error:", e)
        dede
       

if __name__ == "__main__":
    main()