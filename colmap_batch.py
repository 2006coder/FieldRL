#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d


def find_exe(exe_name: str):
    from shutil import which
    return which(exe_name)


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def load_intrinsics(npy_path: Path):
    K = np.load(npy_path)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, got {K.shape}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return K, fx, fy, cx, cy


def list_files_sorted(dir_path: Path, exts):
    files = []
    for ext in exts:
        files.extend(glob.glob(str(dir_path / f"*{ext}")))
    files = sorted(files)
    return [Path(f) for f in files]


def rotmat_to_qvec(R):
    # Returns q = [qw, qx, qy, qz], matching COLMAP convention.
    # Robust conversion based on the trace method.
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    # COLMAP expects qw >= 0 (optional normalization to a canonical hemisphere)
    if q[0] < 0:
        q = -q
    return q


def write_cameras_txt(path: Path, cam_id: int, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    # Write a single PINHOLE camera shared by all images
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{cam_id} PINHOLE {width} {height} {fx:.9f} {fy:.9f} {cx:.9f} {cy:.9f}\n")


def write_images_txt(path: Path, image_names, T_c2w_list, cam_id: int, points2d_per_image=None):
    # COLMAP images.txt stores world-to-camera pose: x_cam = R * x_world + t
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (name, T_c2w) in enumerate(zip(image_names, T_c2w_list), start=1):
            T = np.asarray(T_c2w, dtype=np.float64)
            assert T.shape == (4, 4)
            R_cw = T[:3, :3]
            t_cw = T[:3, 3]
            # world-to-camera
            R = R_cw.T
            t = -R @ t_cw
            q = rotmat_to_qvec(R)
            f.write(f"{i} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {cam_id} {name}\n")
            # 2D points with POINT3D_IDs
            if points2d_per_image is None:
                f.write("\n")
            else:
                pts2d = points2d_per_image[i-1]
                if len(pts2d) == 0:
                    f.write("\n")
                else:
                    line = " ".join([f"{x:.6f} {y:.6f} {pid}" for (x, y, pid) in pts2d])
                    f.write(line + "\n")


def write_points3D_txt(path: Path, points3D=None):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        if not points3D:
            return
        for p in points3D:
            pid = p["id"]
            X, Y, Z = p["xyz"]
            R, G, B = p["rgb"]
            err = p.get("error", 0.0)
            # TRACK: sequence of image_id point2D_idx pairs
            track_pairs = []
            for (image_id, p2d_idx) in p["track"]:
                track_pairs.append(f"{image_id} {p2d_idx}")
            track_str = " ".join(track_pairs)
            f.write(f"{pid} {X:.9f} {Y:.9f} {Z:.9f} {int(R)} {int(G)} {int(B)} {err:.6f}")
            if track_str:
                f.write(f" {track_str}")
            f.write("\n")


def convert_txt_to_bin(txt_dir: Path, bin_dir: Path):
    ensure_dir(bin_dir)
    colmap = find_exe("colmap")
    if not colmap:
        print("WARN: COLMAP not found in PATH. Skipping TXT→BIN conversion.", file=sys.stderr)
        return False

    # Try multiple flag variants for different COLMAP versions
    cmd_variants = [
        [colmap, "model_converter",
         "--input_path", str(txt_dir),
         "--output_path", str(bin_dir),
         "--input_type", "TXT",
         "--output_type", "BIN"],
        [colmap, "model_converter",
         "--input_path", str(txt_dir),
         "--output_path", str(bin_dir),
         "--output_type", "BIN"],  # some versions auto-detect input
        [colmap, "model_converter",
         "--input_path", str(txt_dir),
         "--output_path", str(bin_dir),
         "--input_format", "TXT",
         "--output_format", "BIN"],  # legacy naming
    ]

    last_err = None
    for cmd in cmd_variants:
        try:
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            msg = e.stderr.strip() if e.stderr else str(e)
            print(f"WARN: model_converter failed with '{cmd[-2:]}' variant: {msg}", file=sys.stderr)
            last_err = e

    if last_err:
        print("ERROR: All model_converter variants failed. Please run 'colmap model_converter --help' to see supported flags.", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate COLMAP output from RGB-D data and poses.")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root folder containing 'calibration/rgb_intrinsics.npy' and 'my_scene/{color,depth,pose}'.")
    parser.add_argument("--scene_name", type=str, default="my_scene",
                        help="Scene subfolder name under data root (default: my_scene).")
    parser.add_argument("--output_dir", type=str, default="scene",
                        help="Output scene directory to create (default: ./scene).")
    parser.add_argument("--max_depth", type=float, default=0.5, help="Depth truncation in meters.")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor (e.g., 1000 for mm).")
    parser.add_argument("--compress_ply", action="store_true", help="Write compressed fused.ply.")
    parser.add_argument("--sample_stride", type=int, default=2,
                        help="Pixel stride for sampling depth to create points3D/tracks.")
    parser.add_argument("--voxel_size", type=float, default=0.0033,
                        help="Voxel size in meters for merging multi-view samples into a single 3D point.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    scene_dir_in = data_root / args.scene_name
    calib_path = data_root / "calibration" / "rgb_intrinsics.npy"

    color_dir = scene_dir_in / "color"
    depth_dir = scene_dir_in / "depth"
    pose_dir = scene_dir_in / "pose"

    color_files = list_files_sorted(color_dir, exts=[".png", ".jpg", ".jpeg"])
    depth_files = list_files_sorted(depth_dir, exts=[".npy"])
    pose_files = list_files_sorted(pose_dir, exts=[".npy"])

    if not color_files:
        raise FileNotFoundError(f"No color images in {color_dir}")
    if not depth_files:
        raise FileNotFoundError(f"No depth npy files in {depth_dir}")
    if not pose_files:
        raise FileNotFoundError(f"No pose npy files in {pose_dir}")

    n = len(color_files)
    if not (len(depth_files) == len(pose_files) == n):
        raise ValueError(f"Mismatched counts: colors={len(color_files)}, depths={len(depth_files)}, poses={len(pose_files)}")

    K, fx, fy, cx, cy = load_intrinsics(calib_path)

    # Read one image to get width/height
    im0 = o3d.io.read_image(str(color_files[0]))
    height = np.asarray(im0).shape[0]
    width = np.asarray(im0).shape[1]

    # Set up Open3D intrinsics
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Build fused point cloud (per your reference loop)
    print("Building fused point cloud...")
    scene = o3d.geometry.PointCloud()
    max_depth = float(args.max_depth)

    # Build observations from RGB-D with voxel merging (world-frame)
    print("Collecting observations per voxel...")
    max_depth = float(args.max_depth)
    voxel_size = float(args.voxel_size)
    if voxel_size <= 0:
        raise ValueError("--voxel_size must be > 0")

    # Store the camera-to-world matrices actually used
    T_c2w_list = []

    # Temporary container of voxelized observations:
    # voxels[(ix,iy,iz)] -> list of tuples: (img_id, x, y, Xw(np.array(3,)), rgb(np.array(3,), uint8))
    voxels = {}

    for i in range(n):
        color_raw = o3d.io.read_image(str(color_files[i]))
        depth_np = np.load(depth_files[i])
        T_c2b = np.load(pose_files[i])  # camera->world(base)
        if T_c2b.shape != (4, 4):
            raise ValueError(f"Pose {pose_files[i]} is not 4x4")

        # Store pose for COLMAP
        T_c2w_list.append(T_c2b.astype(np.float64))

        color_img = np.asarray(color_raw)
        h, w = color_img.shape[0], color_img.shape[1]
        z_m = depth_np.astype(np.float64) / float(args.depth_scale)

        R_cw = T_c2b[:3, :3].astype(np.float64)
        t_cw = T_c2b[:3, 3].astype(np.float64)

        stride = max(1, int(args.sample_stride))
        img_id = i + 1  # 1-based for COLMAP

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                z = z_m[y, x]
                if not np.isfinite(z) or z <= 0.0 or z > max_depth:
                    continue
                Xc = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64)
                Xw = R_cw @ Xc + t_cw
                key = tuple(np.floor(Xw / voxel_size).astype(np.int64))
                rgb = np.array(color_img[y, x][:3], dtype=np.uint8)

                lst = voxels.get(key)
                if lst is None:
                    voxels[key] = [(img_id, float(x), float(y), Xw, rgb)]
                else:
                    lst.append((img_id, float(x), float(y), Xw, rgb))

        print(f"Processed frame {i+1}/{n} (samples accumulated)")

    # Build COLMAP data from voxels: averaged points and multi-view tracks
    print(f"Clustering {sum(len(v) for v in voxels.values())} samples into voxels...")
    points2d_per_image = [[] for _ in range(n)]  # list of (x, y, POINT3D_ID) per image
    points3D = []
    next_pid = 1

    # Deterministic ordering of points
    for key in sorted(voxels.keys()):
        obs = voxels[key]
        if not obs:
            continue

        # Deduplicate to at most one observation per image in this voxel
        by_img = {}
        for (img_id, x, y, Xw, rgb) in obs:
            if img_id not in by_img:
                by_img[img_id] = (img_id, x, y, Xw, rgb)
        kept = list(by_img.values())

        # Average XYZ and color across kept observations
        xyzs = np.stack([o[3] for o in kept], axis=0)  # (m,3)
        rgbs = np.stack([o[4] for o in kept], axis=0).astype(np.float64)  # (m,3)
        Xw_avg = xyzs.mean(axis=0)
        rgb_avg = np.clip(np.round(rgbs.mean(axis=0)), 0, 255).astype(np.uint8)

        pid = next_pid
        next_pid += 1

        track = []
        for (img_id, x, y, _Xw, _rgb) in kept:
            p2d_idx = len(points2d_per_image[img_id - 1])
            points2d_per_image[img_id - 1].append((x, y, pid))
            track.append((img_id, p2d_idx))

        points3D.append({
            "id": pid,
            "xyz": (float(Xw_avg[0]), float(Xw_avg[1]), float(Xw_avg[2])),
            "rgb": (int(rgb_avg[0]), int(rgb_avg[1]), int(rgb_avg[2])),
            "error": 0.0,
            "track": track,
        })

    print(f"Generated {len(points3D)} voxel-merged 3D points.")

    # Create fused point cloud from voxel-averaged points (less grainy)
    scene = o3d.geometry.PointCloud()
    if points3D:
        pts = np.array([p["xyz"] for p in points3D], dtype=np.float64)
        cols = np.array([p["rgb"] for p in points3D], dtype=np.float64) / 255.0
        scene.points = o3d.utility.Vector3dVector(pts)
        scene.colors = o3d.utility.Vector3dVector(cols)

    # Prepare output layout
    out_root = Path(args.output_dir).resolve()
    ensure_dir(out_root)

    # 1) Copy images to scene/images
    images_out = out_root / "images"
    ensure_dir(images_out)
    image_names = []
    print("Copying images to scene/images ...")
    for src in color_files:
        name = src.name  # keep original file name (e.g., 000000.png)
        image_names.append(name)
        dst = images_out / name
        if not dst.exists():
            shutil.copyfile(src, dst)

    # 2) Target structure:
    # - scene/sparse/0/*.bin (BIN only)
    # - scene/{n}_views/sparse/0/*.bin (BIN only)
    # - scene/{n}_views/triangulated/*.{txt,bin}
    # - scene/{n}_views/dense/fused.ply
    n_views_dir = out_root / f"{n}_views"
    ensure_dir(n_views_dir)

    sparse_main_bin_dir = out_root / "sparse" / "0"
    sparse_nv_bin_dir = n_views_dir / "sparse" / "0"
    tri_nv_dir = n_views_dir / "triangulated"

    ensure_dir(sparse_main_bin_dir)
    ensure_dir(sparse_nv_bin_dir)
    ensure_dir(tri_nv_dir)

    # We will generate TXT in temporary dirs for sparse, then convert to BIN into the final /sparse/0.
    tmp_sparse_main_txt = out_root / ".tmp_sparse_main_txt"
    tmp_sparse_nv_txt = out_root / ".tmp_sparse_nv_txt"
    # For triangulated, write TXT directly into final folder (we keep both TXT and BIN there).
    # Common camera id
    CAM_ID = 1

    # 2a) Write sparse TXT (temporary) and convert to BIN into sparse/0 (no TXT left in sparse)
    try:
        # main sparse -> BIN at scene/sparse/0
        ensure_dir(tmp_sparse_main_txt)
        write_cameras_txt(tmp_sparse_main_txt / "cameras.txt", CAM_ID, width, height, fx, fy, cx, cy)
        write_images_txt(tmp_sparse_main_txt / "images.txt", image_names, T_c2w_list, CAM_ID, points2d_per_image)
        write_points3D_txt(tmp_sparse_main_txt / "points3D.txt", points3D)
        convert_txt_to_bin(tmp_sparse_main_txt, sparse_main_bin_dir)
    finally:
        # cleanup temp txt dir
        if tmp_sparse_main_txt.exists():
            shutil.rmtree(tmp_sparse_main_txt, ignore_errors=True)

    try:
        # n_views sparse -> BIN at scene/n_views/sparse/0
        ensure_dir(tmp_sparse_nv_txt)
        write_cameras_txt(tmp_sparse_nv_txt / "cameras.txt", CAM_ID, width, height, fx, fy, cx, cy)
        write_images_txt(tmp_sparse_nv_txt / "images.txt", image_names, T_c2w_list, CAM_ID, points2d_per_image)
        write_points3D_txt(tmp_sparse_nv_txt / "points3D.txt", points3D)
        convert_txt_to_bin(tmp_sparse_nv_txt, sparse_nv_bin_dir)
    finally:
        if tmp_sparse_nv_txt.exists():
            shutil.rmtree(tmp_sparse_nv_txt, ignore_errors=True)

    # 2b) Write triangulated TXT directly into scene/n_views/triangulated and convert there (keep TXT and BIN)
    write_cameras_txt(tri_nv_dir / "cameras.txt", CAM_ID, width, height, fx, fy, cx, cy)
    write_images_txt(tri_nv_dir / "images.txt", image_names, T_c2w_list, CAM_ID, points2d_per_image)
    write_points3D_txt(tri_nv_dir / "points3D.txt", points3D)
    try:
        convert_txt_to_bin(tri_nv_dir, tri_nv_dir)
    except subprocess.CalledProcessError as e:
        print(f"WARN: model_converter failed for n_views/triangulated: {e}", file=sys.stderr)

    # 3) Save fused point cloud under scene/{n}_views/dense/fused.ply
    dense_out = n_views_dir / "dense"
    ensure_dir(dense_out)
    fused_path = dense_out / "fused.ply"

    # Ensure normals exist so PLY has nx, ny, nz fields
    if not scene.has_normals() and len(scene.points) > 0:
        print("Estimating normals for fused point cloud...")
        bbox = scene.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(bbox.get_extent()))
        radius = max(1e-3, (0.02 * diag) if diag > 0 else 0.02)
        scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        scene.normalize_normals()
        scene.orient_normals_towards_camera_location(bbox.get_center())

    print(f"Writing fused point cloud to {fused_path}")
    o3d.io.write_point_cloud(str(fused_path), scene, write_ascii=False, compressed=args.compress_ply)

    # Do not create a numeric-only folder or write an 'n_views' file
    print("\nDone.")
    print(f"- Images: {images_out}")
    print(f"- Sparse BIN (main): {sparse_main_bin_dir}")
    print(f"- Sparse BIN ({n}_views): {sparse_nv_bin_dir}")
    print(f"- Triangulated (TXT+BIN): {tri_nv_dir}")
    print(f"- Fused: {fused_path}")


if __name__ == "__main__":
    main()