import torch
import argparse
import os
import numpy as np
from scipy.spatial import cKDTree
import time
from collections import deque, defaultdict

def _find_main_component(positions, eps):
    """
    An extremely fast filtering method that finds the largest connected component
    of the point cloud using a flood-fill (BFS) approach.
    """
    print("Method: Using Ultra-Fast Main Component Analysis...")

    # 1. Build the k-d tree for the entire dataset. This is fast.
    print(f"  - Building k-d tree for {len(positions)} points...")
    kdtree = cKDTree(positions)

    # 2. Find a guaranteed seed point inside the main cluster using voxel hashing.
    # This avoids starting the flood-fill from an outlier.
    print("  - Finding a dense region to start the search...")
    voxel_size = eps * 5.0
    p_min = positions.min(axis=0)
    voxel_indices_list = np.floor((positions - p_min) / voxel_size).astype(int)
    
    # We need to map from a voxel key back to the points within it.
    voxel_map = defaultdict(list)
    for i, key in enumerate(voxel_indices_list):
        voxel_map[tuple(key)].append(i)
        
    if not voxel_map:
        print("Warning: Voxel hashing resulted in no points. Something is wrong.")
        return np.ones(len(positions), dtype=bool)
        
    # Find the densest voxel and pick a random point from it as our seed.
    core_voxel_key = max(voxel_map, key=lambda k: len(voxel_map[k]))
    seed_index = voxel_map[core_voxel_key][0]

    # 3. Perform the Breadth-First Search (BFS) flood-fill.
    print("  - Performing flood-fill search from the seed point...")
    num_points = len(positions)
    is_kept = np.zeros(num_points, dtype=bool)
    queue = deque([seed_index])
    
    # Mark the seed point as already visited/kept
    is_kept[seed_index] = True
    
    while queue:
        # Get the next point to process
        current_index = queue.popleft()
        
        # Find all its neighbors within the 'eps' radius
        neighbors = kdtree.query_ball_point(positions[current_index], r=eps)
        
        # For each neighbor...
        for neighbor_index in neighbors:
            # ...if we haven't already kept it...
            if not is_kept[neighbor_index]:
                # ...mark it as kept and add it to the queue to process later.
                is_kept[neighbor_index] = True
                queue.append(neighbor_index)
                
    return is_kept

def crop_gaussian_cloud(input_path, output_path, eps):
    """
    Loads and crops a Gaussian checkpoint by keeping only the largest connected component.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Loading checkpoint from '{input_path}'...")
    loaded_checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model_data_tuple, iteration = loaded_checkpoint[0], loaded_checkpoint[1]
    
    positions_tensor = model_data_tuple[1]
    initial_count = positions_tensor.shape[0]
    positions_np = positions_tensor.data.cpu().numpy().astype(np.float32) 
    print(f"Loaded {initial_count} total Gaussians.")
    
    start_time = time.time()
    
    # --- The new, fast algorithm is now the only method ---
    is_kept = _find_main_component(positions_np, eps)

    end_time = time.time()
    print(f"Filtering took {end_time - start_time:.2f} seconds.")

    # --- Create new checkpoint from filtered data ---
    kept_count = np.sum(is_kept)
    if kept_count == 0:
        print("Error: Filtering removed all points! Try a larger 'eps' value.")
        return
    
    print(f"\nKept {kept_count} Gaussians ({((kept_count/initial_count)*100):.2f}%).")
    print(f"Discarded {initial_count - kept_count} outlier Gaussians.")

    mask = torch.from_numpy(is_kept)
    new_model_data = []
    for item in model_data_tuple:
        if isinstance(item, torch.Tensor) and item.shape[0] == initial_count:
            new_model_data.append(item[mask])
        else:
            new_model_data.append(item)
    
    new_checkpoint = (tuple(new_model_data), iteration)
    
    print(f"\nSaving cropped checkpoint to '{output_path}'...")
    torch.save(new_checkpoint, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input .pth checkpoint file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the new, cropped .pth file.")
    # We no longer need min_samples. The logic is purely based on distance.
    parser.add_argument("--eps", type=float, required=True, help="The connection distance. Points farther than this from the main component will be discarded. Use find_eps.py to find a good value.")
    
    args = parser.parse_args()
    crop_gaussian_cloud(args.input, args.output, args.eps)

