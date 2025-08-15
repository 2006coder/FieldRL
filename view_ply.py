import open3d as o3d
import os

def view_and_count_ply(file_path):
    """
    Loads a .ply file, prints the number of points, and displays the point cloud.

    Args:
        file_path (str): The full path to the .ply file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # Read the point cloud from the .ply file
        point_cloud = o3d.io.read_point_cloud(file_path)

        # Count the number of points in the point cloud
        num_points = len(point_cloud.points)
        print(f"The .ply file contains {num_points} points.")

        # Visualize the point cloud
        print("Displaying the point cloud. Close the window to exit.")
        o3d.visualization.draw_geometries([point_cloud])

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace this with the actual path to your .ply file
    # For example: "C:/Users/YourUser/Desktop/my_model.ply" on Windows
    # or "/home/user/models/my_model.ply" on Linux
    ply_file = "scene/7_views/dense/fused.ply"

    view_and_count_ply(ply_file)