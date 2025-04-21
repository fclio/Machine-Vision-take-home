import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pyvista as pv

def visualize_centered_points(points_centered, centroid):
    """
    Visualize a centered point cloud with the original centroid marked.
    """
    pcd_centered = o3d.geometry.PointCloud()
    pcd_centered.points = o3d.utility.Vector3dVector(points_centered)
    pcd_centered.paint_uniform_color([0.5, 0.5, 0.5])  # Light gray

    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    origin_sphere.paint_uniform_color([1, 0, 0])  # Red
    origin_sphere.translate([0, 0, 0])

    o3d.visualization.draw_geometries([pcd_centered, origin_sphere])
    print(f"Original centroid: {centroid}")
    print(f"New centroid after centering: {np.mean(points_centered, axis=0)}")


def visualize_pca_and_brick_axes(pcd, pca_axes, brick_dimensions, centroid):
    """
    Visualize PCA axes and compare them to known brick axes.
    """
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    arrow_length = 0.1
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pca_arrows = []

    for i in range(3):
        start = centroid
        end = centroid + pca_axes[i] * arrow_length
        arrow = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([start, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        arrow.paint_uniform_color(colors[i])
        pca_arrows.append(arrow)

    # Brick axes visualization
    brick_colors = [[1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1]]
    brick_arrows = []
    for i in range(3):
        vec = np.zeros(3)
        vec[i] = brick_dimensions[i]
        arrow = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([centroid, centroid + vec]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        arrow.paint_uniform_color(brick_colors[i])
        brick_arrows.append(arrow)

    centroid_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid_point.translate(centroid)
    centroid_point.paint_uniform_color([1, 1, 0])  # Yellow

    o3d.visualization.draw_geometries([pcd] + pca_arrows + brick_arrows + [centroid_point])


def visualize_point_cloud(pcd):
    """
    Display the point cloud with coordinate axes and bounding box.
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)
    o3d.visualization.draw_geometries([pcd, axis, bbox])


def visualize_brick_with_pose_and_pcd(translation, roll, pitch, yaw, pcd, brick_dimensions=(210, 100, 50)):
    """
    Render a 3D brick model centered at (0, 0, 0) and aligned with the estimated pose.
    """
    brick_m = np.array(brick_dimensions) / 1000.0
    box = o3d.geometry.TriangleMesh.create_box(*brick_m)
    box.paint_uniform_color([1.0, 0.647, 0.0])  # RGB for orange

    # Center the brick at the origin
    box.translate(-brick_m / 2.0)

    # Apply rotation
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    box.rotate(r.as_matrix(), center=(0, 0, 0))

    # Offset along Y-axis (brick center)
    box.translate([0, brick_m[1] / 2.0, brick_m[2] / 2.0])

    # Apply estimated translation
    box.translate(np.array(translation) / 1000.0)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([box, coordinate_frame, pcd])


def visualize_pose(intrinsics, translation, rotation, brick_dims = [210, 100, 50]):
    """
    Visualize the camera and brick in PyVista using the estimated pose.
    """
    plotter = pv.Plotter()

    # Origin marker
    origin_dot = pv.Sphere(radius=5, center=(0, 0, 0))
    plotter.add_mesh(origin_dot, color="red")

    # Camera box visualization
    camera_width = intrinsics["width"]
    camera_height = intrinsics["height"]
    camera_box = pv.Cube(bounds=[
        -camera_width / 2, camera_width / 2,
        -1, 1,
        -camera_height / 2, camera_height / 2
    ])
    plotter.add_mesh(camera_box, color="green")

    # Brick center dot
    brick_dot = pv.Sphere(radius=5, center=translation)
    plotter.add_mesh(brick_dot, color="blue")

    # Brick geometry
    brick = pv.Cube(bounds=[
        -brick_dims[0] / 2, brick_dims[0] / 2,
        -brick_dims[1] / 2, brick_dims[1] / 2,
        -brick_dims[2] / 2, brick_dims[2] / 2
    ])

    # Apply rotation (Euler)
    tr_brick = brick.copy()
    brick_center_offset = np.array([0, brick_dims[1] / 2, brick_dims[2] / 2])
    tr_brick = tr_brick.rotate_x(rotation[0])
    tr_brick = tr_brick.rotate_y(rotation[1])
    tr_brick = tr_brick.rotate_z(rotation[2])
    tr_brick = tr_brick.translate(translation + brick_center_offset)

    plotter.add_mesh(tr_brick, color="orange", opacity=0.7)
    plotter.set_background("white")
    plotter.add_axes(line_width=5)
    plotter.show()
