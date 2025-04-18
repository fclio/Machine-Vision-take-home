from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Tuple
import numpy as np
import cv2
import json
import uvicorn
from scipy.spatial.transform import Rotation as R

app = FastAPI()

# Allow cross-origin requests (e.g., from browser frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convert depth image to 3D point cloud using camera intrinsics
def depth_to_point_cloud(depth_img, intrinsics):
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))  # pixel coordinates

    # Depth is stored in 0.1mm units → convert to meters
    z = depth_img / 10000.0

    # Back-project pixel coordinates (u, v) + depth to 3D (x, y, z)
    x = (i - intrinsics['px']) * z / intrinsics['fx']
    y = (j - intrinsics['py']) * z / intrinsics['fy']
    xyz = np.stack((x, y, z), axis=-1)

    # Filter out invalid depth pixels
    mask = z > 0
    return xyz[mask]

# Estimate pose (translation + rotation) using PCA
def estimate_pose_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Get the center of the point cloud → translation of the object
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered = points - centroid

    # Apply SVD to get principal axes (PCA)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    # vh.T gives us the rotation matrix (3x3)
    rot = vh.T
    return centroid, rot

@app.post("/pose")
async def estimate_pose(
    color: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...)
):
    # Load RGB image (unused here but can be for visualization)
    color_img = cv2.imdecode(np.frombuffer(await color.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load depth image (16-bit PNG)
    depth_img = cv2.imdecode(np.frombuffer(await depth.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Load camera intrinsics from cam.json
    intrinsics = json.loads((await camera.read()).decode())

    # Crop center region of the depth image (assuming brick is centered)
    h, w = depth_img.shape
    ch, cw = h // 2, w // 2
    crop_size = 100
    cropped = depth_img[ch - crop_size:ch + crop_size, cw - crop_size:cw + crop_size]
    cv2.imwrite("cropped_depth.png", cropped)
    # Adjust intrinsics for the cropped image
    cam_crop = {
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "px": intrinsics["px"] - (cw - crop_size),
        "py": intrinsics["py"] - (ch - crop_size),
    }

    # Convert cropped depth image to 3D points
    pts = depth_to_point_cloud(cropped, cam_crop)

    # Make sure we have enough points to estimate a stable pose
    if len(pts) < 100:
        return JSONResponse(status_code=400, content={"error": "Too few valid depth points."})

    # Estimate 3D translation (centroid) and orientation (rotation matrix)
    centroid, rot_matrix = estimate_pose_pca(pts)

    # Convert rotation matrix to roll-pitch-yaw angles (in degrees)
    r = R.from_matrix(rot_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)

    # Return translation in millimeters and rotation in degrees
    return {
        "translation": [float(x * 1000) for x in centroid],  # Convert m → mm
        "rotation": [roll, pitch, yaw]
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
