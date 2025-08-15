import os
import sys
import numpy as np

# Add paths to import the LLFF modules
sys.path.append('../../..')  # Path to FGSG-main
from poses.colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary
from poses.colmap_read_model import read_cameras_text, read_images_text, read_points3D_text
from poses.pose_utils import load_colmap_data, save_poses

def process_scene():
    """Process COLMAP scene to create poses_bounds.npy"""
    print("Loading COLMAP data...")
    poses, pts3d, perm = load_colmap_data('.')
    
    print("Saving poses...")
    save_poses('.', poses, pts3d, perm)
    
    print("Done! poses_bounds.npy created.")

if __name__ == "__main__":
    process_scene()