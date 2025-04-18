import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pyvista as pv
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


def depth_to_point_cloud(depth_img, intrinsics):
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img / 10000.0  # Convert 0.1mm to meters
    x = (i - intrinsics['px']) * z / intrinsics['fx']
    y = (j - intrinsics['py']) * z / intrinsics['fy']
    xyz = np.stack((x, y, z), axis=-1)
    mask = z > 0
    return xyz[mask]

def estimate_pose_pca(points: np.ndarray):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rot = vh.T
    return centroid, rot

def run_pose_estimation_from_folder(folder_path: str):
    color_path = os.path.join(folder_path, "color.png")
    depth_path = os.path.join(folder_path, "depth.png")
    cam_json_path = os.path.join(folder_path, "cam.json")

    # Load depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ValueError("Failed to load depth image.")
    h, w = depth_img.shape
    ch, cw = h // 2, w // 2
    crop_size = 100
    cropped = depth_img[ch - crop_size:ch + crop_size, cw - crop_size:cw + crop_size]

    # Save cropped image to check manually
    cv2.imwrite("cropped_depth_debug.png", cropped)

    # Load intrinsics and adjust for crop
    with open(cam_json_path, "r") as f:
        intrinsics = json.load(f)
    cam_crop = {
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "px": intrinsics["px"] - (cw - crop_size),
        "py": intrinsics["py"] - (ch - crop_size),
    }

    # Convert depth to 3D points
    pts = depth_to_point_cloud(cropped, cam_crop)
    if len(pts) < 100:
        print("Too few valid points!")
        return None

    # Estimate pose
    centroid, rot_matrix = estimate_pose_pca(pts)
    r = R.from_matrix(rot_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    
    visualize_pose(depth_img, intrinsics, [float(x * 1000) for x in centroid],[roll, pitch, yaw])

    return {
        "translation_mm": [float(x * 1000) for x in centroid],
        "rotation_deg": [roll, pitch, yaw]
    }

if __name__ == "__main__":
    data_folder = "./place_quality_inputs/0"  # Change this to your actual folder path
    result = run_pose_estimation_from_folder(data_folder)
    
    if result:
        print(json.dumps(result, indent=2))
        print("Cropped depth saved to: cropped_depth_debug.png")
