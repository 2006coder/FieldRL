import torch
import argparse
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def analyze_eps(input_path, k):
    """
    Analyzes a Gaussian checkpoint to help find an optimal eps value for DBSCAN.
    It generates a k-distance graph.
    """
    print(f"Loading checkpoint from '{input_path}'...")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    loaded_checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model_data_tuple = loaded_checkpoint[0]
    positions_tensor = model_data_tuple[1]
    positions_np = positions_tensor.data.cpu().numpy()
    print(f"Loaded {len(positions_np)} Gaussians.")

    print(f"\nCalculating distance to the {k}-th nearest neighbor for all points...")
    # k+1 because the point itself is the 0-th neighbor
    neighbors = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    neighbors_fit = neighbors.fit(positions_np)
    distances, indices = neighbors_fit.kneighbors(positions_np)

    # Get the distance to the k-th neighbor (column k)
    k_distances = np.sort(distances[:, k], axis=0)

    print("Generating k-distance plot... Look for the 'elbow' or 'knee' of the curve.")
    print("This point indicates the optimal 'eps' value.")

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title("k-Distance Graph for Estimating DBSCAN Epsilon (eps)")
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helps find an optimal 'eps' value for DBSCAN by plotting the k-distance graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input .pth checkpoint file.")
    parser.add_argument("--k", type=int, default=10, help="The 'k' value, which should be similar to the 'min_samples' you plan to use for cropping.")
    
    args = parser.parse_args()
    analyze_eps(args.input, args.k)