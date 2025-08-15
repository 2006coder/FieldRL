import torch
import argparse
import os
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

class MiniGaussianModel:
    """A minimal class to load xyz and opacity from a saved Gaussian model."""
    def __init__(self):
        self._xyz = torch.empty(0)
        self._opacity = torch.empty(0)

    def restore_from_captured_tuple(self, model_tuple):
        """Restores parameters from the specific tuple structure from the training script."""
        if not isinstance(model_tuple, tuple) or len(model_tuple) < 7:
            raise TypeError("Loaded model is not the expected tuple structure.")
        self._xyz = model_tuple[1].data
        self._opacity = model_tuple[6].data
        print("Successfully restored 'xyz' and 'opacity' from model tuple.")

    @property
    def get_xyz(self):
        return self._xyz.cpu().numpy()

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity).cpu().numpy()

def visualize_inverse_distance_from_pth(args):
    """
    Visualizes a density field using Inverse Distance Weighting from the nearest neighbor.
    This method is fast, memory-efficient, and requires no distance limit parameter.
    """
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'")
        return

    print(f"Loading checkpoint from '{args.checkpoint}'...")
    loaded_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_data_tuple = loaded_checkpoint[0]
    gaussians = MiniGaussianModel()
    gaussians.restore_from_captured_tuple(model_data_tuple)
    gaussian_positions = gaussians.get_xyz
    gaussian_opacities = gaussians.get_opacity
    print(f"Loaded {len(gaussian_positions)} Gaussians for visualization.")

    print("\nBuilding k-d tree for fast nearest neighbor search...")
    kdtree = cKDTree(gaussian_positions)

    print("Defining a 3D grid around the scene...")
    p_min, p_max = gaussian_positions.min(axis=0), gaussian_positions.max(axis=0)
    padding = (p_max - p_min).max() * 0.05 
    min_bound, max_bound = p_min - padding, p_max + padding

    grid_res = args.grid_resolution
    grid_x, grid_y, grid_z = (np.linspace(min_bound[i], max_bound[i], grid_res) for i in range(3))
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    print(f"Querying {len(grid_points)} grid points for their single nearest neighbor...")
    distances_sq, neighbor_indices = kdtree.query(grid_points, k=1, workers=-1)
    #INVERSE DISTANCE WEIGHTING (IDW) ---

    # 1. Get the base opacity from the nearest neighbor.
    base_opacities = gaussian_opacities[neighbor_indices].flatten()
    
    # 2. Define a small epsilon to prevent division by zero.
    epsilon = 1e-6
    
    # 3. Final density is Opacity / (distance^2 + epsilon).
    # This formula creates a natural falloff without any cutoff distance.
    grid_densities = base_opacities / (distances_sq + epsilon)

    # --- DIAGNOSTICS AND VISUALIZATION ---
    print("\n--- Density Calculation Report ---")
    if grid_densities.max() == 0:
        print("Error: All densities are zero. This is highly unusual.")
        return
    print(f"Max Density Value: {grid_densities.max():.2f}, Mean Density Value: {grid_densities.mean():.2f}")
    print(f"Filtering with density_threshold > {args.density_threshold}")

    visible_mask = grid_densities > args.density_threshold
    if not np.any(visible_mask):
        print("\nNo points passed density threshold. The calculated density values are very different with this method.")
        print("Look at the 'Max Density' and choose a '-d' value within that range (e.g., 5.0, 10.0, 50.0).")
        return
        
    visible_points, visible_densities = grid_points[visible_mask], grid_densities[visible_mask]
    print(f"Visualizing {len(visible_points)} density points.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visible_points)
    
    # Use a logarithmic scale for coloring to better visualize the wide range of density values from IDW.
    log_densities = np.log(visible_densities)
    min_val, max_val = log_densities.min(), log_densities.max()
    norm_densities = (log_densities - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(log_densities)
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("inferno")(norm_densities)[:, :3])
    o3d.visualization.draw_geometries([pcd], window_name="Inverse Distance Weighting Field")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the (cropped) .pth checkpoint file.")
    parser.add_argument("-res", "--grid_resolution", type=int, default=256, help="Resolution of the 3D grid.")
    # The default threshold is now higher as IDW produces larger density values.
    parser.add_argument("-d", "--density_threshold", type=float, default=15.0, help="Minimum density to make a point visible. Adjust based on report.")
    
    args = parser.parse_args()
    visualize_inverse_distance_from_pth(args)