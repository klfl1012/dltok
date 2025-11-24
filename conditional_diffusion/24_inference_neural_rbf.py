#!/usr/bin/env python3
"""
Inference script for Neural RBF.
Visualizes RBF interpolation, CNN refinement, and learned RBF centers.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import importlib.util

current_dir = Path(__file__).parent

# Load Neural RBF model
spec = importlib.util.spec_from_file_location("model_rbf", current_dir / "21_model_neural_rbf.py")
model_rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_rbf)
NeuralRBF = model_rbf.NeuralRBF


def visualize_rbf_centers(model, probe_coords, spatial_size, output_path, var_names):
    """Visualize learned RBF center positions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(var_names):
        ax = axes[i]
        
        # Get RBF centers for this variable's layer
        centers = model.rbf_layers[i].rbf_centers.detach().cpu().numpy()  # [num_rbf_centers, 2]
        
        # Plot RBF centers
        ax.scatter(centers[:, 0] * spatial_size, centers[:, 1] * spatial_size, 
                  c='red', s=50, alpha=0.6, marker='x', label='RBF Centers')
        
        # Plot probe locations
        ax.scatter(probe_coords[:, 0], probe_coords[:, 1], 
                  c='blue', s=30, alpha=0.8, marker='o', label='Probes')
        
        ax.set_xlim(0, spatial_size)
        ax.set_ylim(0, spatial_size)
        ax.set_aspect('equal')
        ax.set_title(f'{var}: Learned RBF Centers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved RBF centers visualization: {output_path}")


@torch.no_grad()
def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--probe-dir", type=str, default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--num-rbf-centers", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-timesteps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/results/neural_rbf")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load probe coordinates
    probe_coords = np.load(Path(args.probe_dir) / "probe_coordinates.npy")
    probe_coords_norm = probe_coords.copy().astype(np.float32)
    probe_coords_norm[:, 0] /= 1024
    probe_coords_norm[:, 1] /= 1024
    probe_coords_t = torch.from_numpy(probe_coords_norm).to(device)
    
    # Create model
    model = NeuralRBF(
        num_probes=len(probe_coords),
        num_vars=len(args.variables),
        num_rbf_centers=args.num_rbf_centers,
        spatial_size=(args.spatial_size, args.spatial_size),
        base_channels=args.base_channels,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Visualize learned RBF centers
    visualize_rbf_centers(model, probe_coords, args.spatial_size, 
                         output_dir / "rbf_centers.png", args.variables)
    
    # Load data
    probe_data = {}
    field_data = {}
    stats = {}
    
    for var in args.variables:
        probe_vals = np.load(Path(args.probe_dir) / f"probe_{var}.npy")
        field_vals = np.load(Path(args.data_dir) / f"{var}.npy")
        
        mean = field_vals.mean()
        std = field_vals.std()
        stats[var] = {'mean': mean, 'std': std}
        
        probe_vals = (probe_vals - mean) / (std + 1e-8)
        field_vals = (field_vals - mean) / (std + 1e-8)
        
        probe_data[var] = probe_vals
        field_data[var] = field_vals
    
    # Use validation set (last 20%)
    total_timesteps = probe_vals.shape[0]
    val_start = int(total_timesteps * 0.8)
    timesteps = np.linspace(val_start, total_timesteps - 1, args.num_timesteps, dtype=int)
    
    print(f"Generating inference for {args.num_timesteps} timesteps...")
    
    # Generate comparisons
    frames_rbf = {var: [] for var in args.variables}
    frames_refined = {var: [] for var in args.variables}
    frames_gt = {var: [] for var in args.variables}
    frames_comparison = []
    
    for t in timesteps:
        # Prepare input
        probe_vals = np.stack([probe_data[var][t] for var in args.variables], axis=0)
        probe_vals_t = torch.from_numpy(probe_vals).unsqueeze(0).float().to(device)
        
        # Inference
        rbf_out, refined_out = model(probe_vals_t, probe_coords_t)
        
        # Process outputs per variable
        for i, var in enumerate(args.variables):
            # Ground truth
            gt = field_data[var][t]
            gt_resized = torch.nn.functional.interpolate(
                torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float(),
                size=(args.spatial_size, args.spatial_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Predictions
            rbf_pred = rbf_out[0, i].cpu().numpy()
            refined_pred = refined_out[0, i].cpu().numpy()
            
            # Denormalize
            mean, std = stats[var]['mean'], stats[var]['std']
            gt_denorm = gt_resized * std + mean
            rbf_denorm = rbf_pred * std + mean
            refined_denorm = refined_pred * std + mean
            
            # Append frames
            vmin, vmax = gt_denorm.min(), gt_denorm.max()
            
            frames_rbf[var].append((rbf_denorm - vmin) / (vmax - vmin + 1e-8))
            frames_refined[var].append((refined_denorm - vmin) / (vmax - vmin + 1e-8))
            frames_gt[var].append((gt_denorm - vmin) / (vmax - vmin + 1e-8))
        
        # Create comparison figure
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        for i, var in enumerate(args.variables):
            gt = frames_gt[var][-1]
            rbf_pred = frames_rbf[var][-1]
            refined_pred = frames_refined[var][-1]
            
            # Ground truth
            axes[i, 0].imshow(gt, cmap='viridis')
            axes[i, 0].set_title(f'{var} - Ground Truth')
            axes[i, 0].axis('off')
            
            # RBF interpolation
            axes[i, 1].imshow(rbf_pred, cmap='viridis')
            mse_rbf = np.mean((rbf_pred - gt) ** 2)
            axes[i, 1].set_title(f'{var} - RBF (MSE={mse_rbf:.4f})')
            axes[i, 1].axis('off')
            
            # Refined output
            axes[i, 2].imshow(refined_pred, cmap='viridis')
            mse_ref = np.mean((refined_pred - gt) ** 2)
            axes[i, 2].set_title(f'{var} - Refined (MSE={mse_ref:.4f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames_comparison.append(frame)
        plt.close()
    
    # Save comparison GIF
    imageio.mimsave(output_dir / "rbf_comparison.gif", frames_comparison, fps=2)
    print(f"Saved comparison: {output_dir / 'rbf_comparison.gif'}")
    
    # Save per-variable GIFs
    for var in args.variables:
        # RBF only
        rbf_frames_uint8 = [(f * 255).astype(np.uint8) for f in frames_rbf[var]]
        imageio.mimsave(output_dir / f"{var}_rbf.gif", rbf_frames_uint8, fps=2)
        
        # Refined only
        ref_frames_uint8 = [(f * 255).astype(np.uint8) for f in frames_refined[var]]
        imageio.mimsave(output_dir / f"{var}_refined.gif", ref_frames_uint8, fps=2)
        
        print(f"Saved {var}: rbf.gif, refined.gif")
    
    print(f"\nInference complete! Results in {output_dir}")


if __name__ == "__main__":
    inference()
