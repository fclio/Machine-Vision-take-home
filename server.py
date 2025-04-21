from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import json
from algorithm import compute_rotation_translation
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

def image_to_np_array(image_data):
    """Convert an uploaded image file to a numpy array."""
    image = Image.open(io.BytesIO(image_data))  # Open the image from the byte stream
    return np.array(image)

def json_to_dict(json_data):
    """Convert uploaded JSON file to a dictionary."""
    return json.loads(json_data)

@app.post("/pose")
async def estimate_pose(
    color: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
):
    # Read the uploaded files and convert to numpy arrays
    color_image_data = await color.read()  # Read color image file
    depth_image_data = await depth.read()  # Read depth image file

    color_image_np = image_to_np_array(color_image_data)  # Convert color image to numpy array
    depth_image_np = image_to_np_array(depth_image_data)  # Convert depth image to numpy array

    # Read the camera intrinsic parameters from the uploaded JSON file
    camera_data = await camera.read()  # Read camera JSON file
    intrinsics = json_to_dict(camera_data)  # Convert JSON data to dictionary

    # Compute translation and rotation
    translation, rotation = compute_rotation_translation(
        depth_image_np, color_image_np, intrinsics, visualize=False
    )

    return {
        "translation": translation,
        "rotation": rotation
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
