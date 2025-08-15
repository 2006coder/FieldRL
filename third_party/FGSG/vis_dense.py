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

def visualize_inverse_distance_batched_from_pth(args):
    """
    Visualizes a density field using batched Inverse Distance Weighting to handle
    extremely high grid resolutions with low memory usage.
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
    
    # We create the axis vectors, but not the full grid yet.
    grid_x = np.linspace(min_bound[0], max_bound[0], grid_res)
    grid_y = np.linspace(min_bound[1], max_bound[1], grid_res)
    grid_z = np.linspace(min_bound[2], max_bound[2], grid_res)
    
    # Lists to store the final results that pass the threshold
    all_visible_points = []
    all_visible_densities = []
    
    epsilon = 1e-6 # To prevent division by zero in IDW

    print(f"\nProcessing grid in {grid_res} slices to conserve memory...")
    
    # --- NEW CORE LOGIC: BATCHING BY SLICE ---
    # We process the grid one X-slice at a time.
    for i in tqdm(range(grid_res), desc="Processing Grid Slices"):
        # 1. Generate grid points for ONLY the current slice
        # This creates a grid_res x grid_res plane of points.
        yy, zz = np.meshgrid(grid_y, grid_z, indexing='ij')
        # All points in this slice have the same x-coordinate
        xx = np.full_like(yy, grid_x[i])
        
        batch_grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # 2. Query k-d tree only for this slice's points
        distances_sq, neighbor_indices = kdtree.query(batch_grid_points, k=1, workers=-1)
        
        # 3. Calculate densities for this slice
        base_opacities = gaussian_opacities[neighbor_indices].flatten()
        batch_densities = base_opacities / (distances_sq + epsilon)
        
        # 4. Filter this slice's points and densities
        visible_mask = batch_densities > args.density_threshold
        
        # 5. Append only the points that passed the filter to our final list
        if np.any(visible_mask):
            all_visible_points.append(batch_grid_points[visible_mask])
            all_visible_densities.append(batch_densities[visible_mask])
            
    # --- END OF BATCHING LOOP ---

    if not all_visible_points:
        print("\nNo points passed the density threshold. Try lowering the '-d' value.")
        return

    # Now, we create the final large arrays from the lists of visible points
    final_visible_points = np.vstack(all_visible_points)
    final_visible_densities = np.concatenate(all_visible_densities)

    print(f"\nVisualizing {len(final_visible_points)} density points.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_visible_points)
    
    # Use a logarithmic scale for coloring to better visualize the wide range of density values
    log_densities = np.log(final_visible_densities)
    min_val, max_val = log_densities.min(), log_densities.max()
    norm_densities = (log_densities - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(log_densities)
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("inferno")(norm_densities)[:, :3])
    
    o3d.visualization.draw_geometries([pcd], window_name="Batched Inverse Distance Field")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a Gaussian checkpoint using batched Inverse Distance Weighting for very high resolutions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the (cropped) .pth checkpoint file.")
    parser.add_argument("-res", "--grid_resolution", type=int, default=300, help="Resolution of the 3D grid. High values are now memory-safe.")
    parser.add_argument("-d", "--density_threshold", type=float, default=4, help="Minimum density to make a point visible. Adjust based on results.")
    
    args = parser.parse_args()
    visualize_inverse_distance_batched_from_pth(args)