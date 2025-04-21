import os
import requests
import json
from visualization import visualize_pose


def send_to_server(color_path, depth_path, cam_json_path, endpoint="http://localhost:8000/pose"):
    with open(color_path, "rb") as c, open(depth_path, "rb") as d, open(cam_json_path, "rb") as j:
        files = {
            "color": ("color.png", c, "image/png"),
            "depth": ("depth.png", d, "image/png"),
            "camera": ("cam.json", j, "application/json"),
        }
        response = requests.post(endpoint, files=files)
        # Ensure server returns a valid response
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Server error: {response.status_code}, {response.text}")


def main():
    dataset_dir = "place_quality_inputs"  # Path to folders with color.png, depth.png, cam.json
    for folder in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(path):
            continue

        print(f"\nProcessing {folder}")
        try:
            # Send data to server and get pose estimate
            result = send_to_server(
                os.path.join(path, "color.png"),
                os.path.join(path, "depth.png"),
                os.path.join(path, "cam.json")
            )

            print("Pose:", json.dumps(result, indent=2))

            # Extract translation and rotation from the result
            translation = result["translation"]
            rotation = result["rotation"]

            # Read camera intrinsics
            with open(os.path.join(path, "cam.json")) as f:
                intrinsics = json.load(f)

            print("Pose Estimation Result:")
            print(f"Translation (x, y, z): {translation}")
            print(f"Rotation (Euler angles):")
            print(f"  Roll (X): {rotation[0]}°")
            print(f"  Pitch (Y): {rotation[1]}°")
            print(f"  Yaw (Z): {rotation[2]}°")

            # Visualize pose
            visualize_pose(intrinsics, translation=translation, rotation=rotation)
            
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
