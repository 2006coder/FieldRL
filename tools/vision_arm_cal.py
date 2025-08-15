import argparse
import os
import time

import cv2
import numpy as np
import rtde_receive
import rtde_control 
from scipy.spatial.transform import Rotation as R


from tools.realsense import RealSense

def collect_data(output_dir, calibration_dir, robot_ip="10.0.0.78"):

    print("Initializing...")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["color", "depth", "pose"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Initialize RealSense camera
    camera = RealSense(align_color=True, structured_light=1)
    print("RealSense camera initialized.")

    # Initialize RTDE connection to the robot
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        # 2. Add control interface to reset the TCP
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        print("Connected to UR robot.")

        # Reset TCP offset to zero to ensure getActualTCPPose() returns the flange pose
        print("Resetting TCP offset to zero...")
        tool_offset_pose = np.zeros(6)
        rtde_c.setTcp(tool_offset_pose)
        time.sleep(0.2)  # Give the controller a moment to process the change
        print("TCP offset has been reset.")

    except RuntimeError as e:
        print(f"Error connecting to the robot at {robot_ip}: {e}")
        camera.release()
        return

    # Load calibration data (camera_to_flange transform)
    try:
        T_c2f = np.load(os.path.join(calibration_dir, 'camera_to_flange.npy'))
        print("Loaded camera_to_flange transform:")
        print(T_c2f)
    except FileNotFoundError:
        print(f"Error: 'camera_to_flange.npy' not found in '{calibration_dir}'.")
        print("Please run the calibration script first.")
        camera.release()
        return

    # Save camera intrinsics once
    intrinsics_path = os.path.join(output_dir, "rgb_intrinsics.npy")
    if not os.path.exists(intrinsics_path):
        K = camera.get_rgb_intrinsics()
        np.save(intrinsics_path, K)
        print(f"Saved camera intrinsics to {intrinsics_path}")
    
    # --- Data Collection Loop ---
    print("\nStarting data collection...")
    print("Press 's' to save a frame, 'q' to quit.")
    
    frame_idx = 0
    while True:
        # Get data from camera
        depth_image, color_image = camera.get_aligned_rgbd()
        if depth_image is None or color_image is None:
            continue

        # Get data from robot
        actual_pose_vec = rtde_r.getActualTCPPose()
        
        # Calculate flange-to-base transform T_f2b
        T_f2b = np.eye(4)
        T_f2b[:3, :3] = R.from_rotvec(actual_pose_vec[3:]).as_matrix()
        T_f2b[:3, 3] = actual_pose_vec[:3]
        
        # Calculate camera-to-base transform: T_c2b = T_f2b @ T_c2f
        T_c2b = T_f2b @ T_c2f

        # Display the live feed
        cv2.imshow("Live RGB Feed", color_image)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            # Define file paths for the current frame
            color_path = os.path.join(output_dir, "color", f"{frame_idx:06d}.png")
            depth_path = os.path.join(output_dir, "depth", f"{frame_idx:06d}.npy")
            pose_path = os.path.join(output_dir, "pose", f"{frame_idx:06d}.npy")

            # Save the data
            cv2.imwrite(color_path, color_image)
            np.save(depth_path, depth_image)
            np.save(pose_path, T_c2b)
            
            print(f"Saved frame {frame_idx}: {color_path}, {depth_path}, {pose_path}")
            frame_idx += 1

        elif key == ord('q'):
            print("Quitting...")
            break
            
    # --- Cleanup ---
    camera.release()
    cv2.destroyAllWindows()
    rtde_c.disconnect() # Disconnect the control interface
    print("Collection finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect RGB-D data and robot poses for 3D reconstruction.")
    parser.add_argument('--output_dir', default='data/my_scene', help='Directory to save the collected data.')
    parser.add_argument('--calibration_dir', default='data/calibration', help='Directory containing calibration files (camera_to_flange.npy).')
    args = parser.parse_args()

    collect_data(args.output_dir, args.calibration_dir)