import os
import torch
import argparse
import numpy as np
import h5py
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams


def export_gaussians_to_npz(ckpt_path, out_path):
    print(f"Loading Gaussian model from {ckpt_path} ...")
    model_path = os.path.dirname(ckpt_path)
    dummy_args = [f'--model_path={model_path}', '--train_bg']
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument('--train_bg', action='store_true', default=False)
    args = parser.parse_args(dummy_args)

    gaussians = GaussianModel(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_state = checkpoint[0]
    else:
        model_state = checkpoint
    gaussians.restore(model_state, args)
    xyz = gaussians.get_xyz.cpu().detach().numpy()
    scales = gaussians.get_scaling.cpu().detach().numpy()
    opacity = torch.sigmoid(gaussians.get_opacity).cpu().detach().numpy()
    if len(opacity.shape) > 1:
        opacity = opacity.flatten()
    print(f"Exporting {xyz.shape[0]} Gaussians to {out_path} ...")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('xyz', data=xyz, compression='gzip')
        f.create_dataset('scales', data=scales, compression='gzip')
        f.create_dataset('opacity', data=opacity, compression='gzip')
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Export Gaussian model parameters to .h5 file")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to chkpnt10000.pth')
    parser.add_argument('--out', type=str, required=True, help='Output .h5 file path')
    args = parser.parse_args()
    export_gaussians_to_npz(args.ckpt, args.out)

if __name__ == "__main__":
    main()
