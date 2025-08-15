colmap feature_extractor --database_path colmap.db --image_path images

colmap exhaustive_matcher --database_path colmap.db

mkdir sparse

colmap mapper --database_path colmap.db --image_path images --output_path sparse

mkdir colmap_text

colmap model_converter --input_path sparse/0 --output_path colmap_text --output_type TXT

colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense --output_type COLMAP

colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.num_iterations 2 --PatchMatchStereo.num_samples 5

colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply

cd dataset/nerf_llff_data/myhouse

python createposebounds.py


