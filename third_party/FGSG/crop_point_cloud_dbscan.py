import os
import numpy as np
import open3d as o3d
import argparse

def crop_point_cloud_dbscan(input_ply, output_ply, eps=0.6, min_samples=1):
    """Crop point cloud using DBSCAN and save the largest cluster to a new PLY file."""
    print(f"Loading point cloud: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64)
    print(f"Total points: {len(xyz)}")
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn"])
        from sklearn.cluster import DBSCAN
    if len(xyz) > 100:
        print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
        labels = db.labels_
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique) > 0:
            main_label = unique[np.argmax(counts)]
            main_mask = labels == main_label
            xyz = xyz[main_mask]
            colors = colors[main_mask] if len(colors) == len(labels) else colors
            print(f"Cropped to main cluster: {len(xyz)} points")
        else:
            print("No clusters found, saving original point cloud.")
    else:
        print("Point cloud too small for DBSCAN, saving original.")
    # Save new point cloud
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
    if len(colors) == len(xyz):
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_ply, cropped_pcd)
    print(f"Exported cropped point cloud to: {output_ply}")


def main():
    parser = argparse.ArgumentParser(description="Crop point cloud using DBSCAN and export main cluster.")
    parser.add_argument('--input', type=str, required=True, help='Input point cloud PLY file')
    parser.add_argument('--output', type=str, required=True, help='Output cropped PLY file')
    parser.add_argument('--eps', type=float, default=0.6, help='DBSCAN eps parameter (default: 0.6)')
    parser.add_argument('--min_samples', type=int, default=1, help='DBSCAN min_samples parameter (default: 1)')
    args = parser.parse_args()
    crop_point_cloud_dbscan(args.input, args.output, eps=args.eps, min_samples=args.min_samples)

if __name__ == "__main__":
    main()
