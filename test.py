import os
import json
import cv2
from algorithm import compute_rotation_translation
from visualization import visualize_pose

def run_pose_estimation_from_folder(folder_path: str, visualize: bool = False):
    """Run pose estimation on a folder containing color, depth, and camera intrinsics."""
    color_path = os.path.join(folder_path, "color.png")
    depth_path = os.path.join(folder_path, "depth.png")
    cam_json_path = os.path.join(folder_path, "cam.json")

    # Read images
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Load camera intrinsics
    with open(cam_json_path, 'r') as f:
        intrinsics = json.load(f)

    # Compute translation and rotation
    translation, rotation = compute_rotation_translation(
        depth_image, color_image, intrinsics, visualize=visualize
    )

    # Visualize pose (if needed)
    if visualize:
        visualize_pose(intrinsics, translation=translation, rotation=rotation)

    return translation, rotation


if __name__ == "__main__":
    dataset_dir = "place_quality_inputs"  # Change this to your actual folder path

    # Process each folder in the dataset directory
    for folder in sorted(os.listdir(dataset_dir)):
        if folder.startswith('.'):
            continue  # Skip hidden files/folders

        folder_path = os.path.join(dataset_dir, folder)
        print(f"\nProcessing folder: {folder}")

        translation, rotation = run_pose_estimation_from_folder(folder_path, visualize=True)

        print(f"Translation: {translation}")
        print(f"Rotation Matrix:\n{rotation}")
