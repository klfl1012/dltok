#!/usr/bin/env python3
"""
VAE inference script for probe-based reconstruction.

Tests two conditioning methods:
1. Sparse probe data with noise background
2. Bilinear interpolation from probe data

Generates GIFs and comparison images for each variable.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

# Import model
import importlib.util

current_dir = Path(__file__).parent

# Load 5_model.py
spec = importlib.util.spec_from_file_location("model_5", current_dir / "5_model.py")
model_5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_5)
VAE = model_5.VAE


def load_model(checkpoint_path: Path, device, num_vars: int = 4):
    """Load trained VAE model from checkpoint."""
    # First load checkpoint to determine the model size
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to infer input size from decoder.fc weight shape
    # decoder.fc projects latent_dim to (base_channels * 8 * spatial_size * spatial_size)
    # For base_channels=32: spatial_size = sqrt(fc_out_features / (32 * 8)) = sqrt(fc_out_features / 256)
    fc_out_features = checkpoint['model_state_dict']['decoder.fc.weight'].shape[0]
    spatial_size = int(np.sqrt(fc_out_features / 256))
    input_size = spatial_size * 16  # Since we downsample 4 times (/16)
    
    print(f"Detected model input size: {input_size}×{input_size}")
    
    model = VAE(
        in_channels=num_vars,
        latent_dim=256,
        base_channels=32,
        input_size=input_size,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model, input_size


def create_probe_image_sparse(
    probe_coords: np.ndarray,
    probe_values: np.ndarray,
    target_size: tuple = (1024, 1024),
    noise_std: float = 0.1,
) -> np.ndarray:
    """
    Create sparse probe image with noise background.
    
    Parameters
    ----------
    probe_coords : np.ndarray
        Probe coordinates, shape (num_probes, 2)
    probe_values : np.ndarray
        Probe values, shape (num_probes,)
    target_size : tuple
        Output image size (H, W)
    noise_std : float
        Standard deviation of background noise
    
    Returns
    -------
    image : np.ndarray
        Sparse probe image, shape (H, W)
    """
    H, W = target_size
    image = np.random.randn(H, W) * noise_std
    
    # Fill in probe values at their locations
    for (x, z), value in zip(probe_coords, probe_values):
        if 0 <= x < W and 0 <= z < H:
            image[z, x] = value
    
    return image


def create_probe_image_interpolated(
    probe_coords: np.ndarray,
    probe_values: np.ndarray,
    target_size: tuple = (1024, 1024),
) -> np.ndarray:
    """
    Create probe image with bilinear interpolation.
    
    For probes on a horizontal line, uses 1D interpolation along x-axis
    and broadcasts to other rows.
    
    Parameters
    ----------
    probe_coords : np.ndarray
        Probe coordinates, shape (num_probes, 2)
    probe_values : np.ndarray
        Probe values, shape (num_probes,)
    target_size : tuple
        Output image size (H, W)
    
    Returns
    -------
    image : np.ndarray
        Interpolated image, shape (H, W)
    """
    from scipy.interpolate import interp1d
    
    H, W = target_size
    
    # Check if all probes are on the same horizontal line
    unique_z = np.unique(probe_coords[:, 1])
    
    if len(unique_z) == 1:
        # Probes are on a horizontal line - use 1D interpolation
        probe_z = unique_z[0]
        probe_x = probe_coords[:, 0]
        
        # Sort by x coordinate
        sort_idx = np.argsort(probe_x)
        probe_x_sorted = probe_x[sort_idx]
        probe_vals_sorted = probe_values[sort_idx]
        
        # Create 1D interpolator along x-axis
        interpolator = interp1d(
            probe_x_sorted,
            probe_vals_sorted,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Interpolate along x-axis
        x_coords = np.arange(W)
        interpolated_line = interpolator(x_coords)
        
        # Create image by broadcasting
        image = np.zeros((H, W))
        image[probe_z, :] = interpolated_line
        
        # Optionally: extend to neighboring rows with decay
        # Simple approach: copy the line to all rows (no vertical structure)
        # Better approach: decay from the probe line
        for z in range(H):
            distance = abs(z - probe_z)
            decay = np.exp(-distance / (H / 10))  # Decay over ~10% of height
            image[z, :] = interpolated_line * decay
        
        return image
    else:
        # Full 2D interpolation
        from scipy.interpolate import griddata
        
        grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(H))
        
        image = griddata(
            probe_coords,
            probe_values,
            (grid_x, grid_z),
            method='linear',
            fill_value=0.0
        )
        
        return image


def normalize_for_display(data: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Normalize data to [0, 1] for visualization."""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    if vmax > vmin:
        return (data - vmin) / (vmax - vmin)
    else:
        return np.zeros_like(data)


def create_comparison_frame(
    ground_truth: np.ndarray,
    sparse_recon: np.ndarray,
    interp_recon: np.ndarray,
    variable_name: str,
    probe_coords: np.ndarray = None,
    cmap: str = 'viridis',
) -> np.ndarray:
    """
    Create comparison frame with ground truth and two reconstructions.
    
    Returns RGB image as numpy array.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Compute shared vmin/vmax for consistent coloring
    vmin = min(ground_truth.min(), sparse_recon.min(), interp_recon.min())
    vmax = max(ground_truth.max(), sparse_recon.max(), interp_recon.max())
    
    # Row 1: Ground truth, Sparse, Interpolated
    axes[0, 0].imshow(ground_truth, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{variable_name} - Ground Truth')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sparse_recon, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'{variable_name} - Sparse Probe Recon')
    axes[0, 1].axis('off')
    
    im = axes[0, 2].imshow(interp_recon, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'{variable_name} - Interpolated Recon')
    axes[0, 2].axis('off')
    
    # Row 2: Errors
    sparse_error = np.abs(ground_truth - sparse_recon)
    interp_error = np.abs(ground_truth - interp_recon)
    error_max = max(sparse_error.max(), interp_error.max())
    
    axes[1, 0].axis('off')  # Empty
    
    axes[1, 1].imshow(sparse_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'Sparse Error (MAE={sparse_error.mean():.4f})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(interp_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 2].set_title(f'Interp Error (MAE={interp_error.mean():.4f})')
    axes[1, 2].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[0, :], fraction=0.046, pad=0.04)
    
    # Overlay probe locations if provided
    if probe_coords is not None:
        for ax in axes[0, :]:
            ax.scatter(probe_coords[:, 0], probe_coords[:, 1], 
                      c='red', s=10, marker='x', alpha=0.5)
    
    plt.tight_layout()
    
    # Convert to numpy array using modern matplotlib API
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]  # Drop alpha channel
    
    plt.close(fig)
    
    return image


@torch.no_grad()
def run_inference(
    model: VAE,
    data_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    variables: list,
    device,
    resize: int = 256,
    noise_std: float = 0.1,
    denoise_iterations: list = [1, 2, 5, 10],
):
    """
    Run inference on probe data and generate visualizations.
    
    Parameters
    ----------
    model : VAE
        Trained VAE model
    data_dir : Path
        Directory with ground truth .npy files
    probe_dir : Path
        Directory with probe data
    output_dir : Path
        Output directory for results
    variables : list
        List of variable names
    device : torch.device
        Device to run on
    resize : int
        Image size for model input
    noise_std : float
        Noise level for sparse probe images
    denoise_iterations : list
        List of iteration counts to test (e.g., [1, 2, 5, 10])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running inference...")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load probe coordinates
    probe_coords = np.load(probe_dir / "probe_coordinates.npy")
    print(f"Probe coordinates: {probe_coords.shape}")
    
    # Load normalization stats (compute from ground truth)
    stats = {}
    gt_data_all = {}
    probe_data_all = {}
    
    for var_name in variables:
        data = np.load(data_dir / f"{var_name}.npy")
        stats[var_name] = {
            'mean': data.mean(),
            'std': data.std()
        }
        gt_data_all[var_name] = data
        probe_data_all[var_name] = np.load(probe_dir / f"probe_{var_name}.npy")
    
    # Get number of timesteps
    num_timesteps = gt_data_all[variables[0]].shape[0]
    print(f"Number of timesteps: {num_timesteps}")
    print(f"Processing all {len(variables)} variables simultaneously")
    print(f"Denoising iterations to test: {denoise_iterations}")
    
    # Storage for all variables' frames and all iteration counts
    # Structure: all_frames[method][var_name][n_iters] = list of frames
    all_gt_frames = {var: [] for var in variables}
    all_sparse_frames = {var: {n: [] for n in denoise_iterations} for var in variables}
    all_interp_frames = {var: {n: [] for n in denoise_iterations} for var in variables}
    
    # Process all timesteps
    for t in range(num_timesteps):
        if t % 10 == 0:
            print(f"  Processing frame {t}/{num_timesteps}")
        
        # Prepare multi-channel inputs for this timestep
        sparse_channels = []
        interp_channels = []
        gt_channels = []
        
        for var_name in variables:
            # Get ground truth
            gt_frame = gt_data_all[var_name][t]  # (1024, 1024)
            
            # Normalize
            gt_normalized = (gt_frame - stats[var_name]['mean']) / (stats[var_name]['std'] + 1e-8)
            gt_channels.append(gt_normalized)
            
            # Get probe values for this timestep
            probe_vals = probe_data_all[var_name][t]  # (num_probes,)
            
            # Create sparse probe image
            sparse_input = create_probe_image_sparse(
                probe_coords,
                probe_vals,
                target_size=gt_frame.shape,
                noise_std=noise_std
            )
            sparse_channels.append(sparse_input)
            
            # Create interpolated probe image
            interp_input = create_probe_image_interpolated(
                probe_coords,
                probe_vals,
                target_size=gt_frame.shape,
            )
            interp_channels.append(interp_input)
        
        # Stack all variables into multi-channel tensors
        sparse_multi = np.stack(sparse_channels, axis=0)  # (num_vars, H, W)
        interp_multi = np.stack(interp_channels, axis=0)  # (num_vars, H, W)
        
        # Convert to torch and resize
        sparse_tensor = torch.from_numpy(sparse_multi).float().unsqueeze(0)  # (1, num_vars, H, W)
        interp_tensor = torch.from_numpy(interp_multi).float().unsqueeze(0)
        
        sparse_resized = F.interpolate(sparse_tensor, size=(resize, resize), mode='bilinear', align_corners=False)
        interp_resized = F.interpolate(interp_tensor, size=(resize, resize), mode='bilinear', align_corners=False)
        
        # Initialize with probe inputs (keep probe data throughout iterations)
        sparse_current = sparse_resized.clone().to(device)
        interp_current = interp_resized.clone().to(device)
        
        # Keep original probe inputs for reinsertion
        sparse_probe_input = sparse_resized.to(device)
        interp_probe_input = interp_resized.to(device)
        
        # Process with different iteration counts
        for n_iters in denoise_iterations:
            # Reset to probe input for this iteration count
            if n_iters == 1:
                # First iteration - use original probe input
                sparse_current = sparse_probe_input.clone()
                interp_current = interp_probe_input.clone()
            # else: continue from previous iteration's output
            
            # Iteratively denoise
            for iter_idx in range(1 if n_iters == 1 else n_iters - denoise_iterations[denoise_iterations.index(n_iters) - 1]):
                sparse_recon, _, _ = model(sparse_current)
                interp_recon, _, _ = model(interp_current)
                
                # Restore probe locations in the reconstruction
                # For sparse: restore probe values at probe pixel locations
                # For interp: restore interpolated probe line
                sparse_current = sparse_recon.clone()
                interp_current = interp_recon.clone()
                
                # Re-inject probe data (preserve probe information)
                # This keeps the probe measurements consistent across iterations
                for var_idx, var_name in enumerate(variables):
                    probe_vals = probe_data_all[var_name][t]
                    
                    # For sparse: put probe values back at exact locations
                    for probe_idx, (x, z) in enumerate(probe_coords):
                        # Scale coordinates to resized image
                        x_scaled = int(x * resize / gt_data_all[var_name][t].shape[1])
                        z_scaled = int(z * resize / gt_data_all[var_name][t].shape[0])
                        
                        if 0 <= x_scaled < resize and 0 <= z_scaled < resize:
                            # Normalize probe value
                            probe_val_norm = (probe_vals[probe_idx] - stats[var_name]['mean']) / (stats[var_name]['std'] + 1e-8)
                            sparse_current[0, var_idx, z_scaled, x_scaled] = probe_val_norm
            
            # Resize back to original resolution and move to CPU
            sparse_recon_full = F.interpolate(
                sparse_current,
                size=gt_data_all[variables[0]][t].shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).cpu().numpy()  # (num_vars, H, W)
            
            interp_recon_full = F.interpolate(
                interp_current,
                size=gt_data_all[variables[0]][t].shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).cpu().numpy()  # (num_vars, H, W)
            
            # Denormalize and store each variable's reconstruction
            for var_idx, var_name in enumerate(variables):
                # Denormalize
                sparse_recon_var = sparse_recon_full[var_idx] * stats[var_name]['std'] + stats[var_name]['mean']
                interp_recon_var = interp_recon_full[var_idx] * stats[var_name]['std'] + stats[var_name]['mean']
                
                # Store
                all_sparse_frames[var_name][n_iters].append(sparse_recon_var)
                all_interp_frames[var_name][n_iters].append(interp_recon_var)
        
        # Clean up
        del sparse_current, interp_current, sparse_probe_input, interp_probe_input
        
        # Store ground truth (only once per timestep)
        for var_name in variables:
            gt_var = gt_data_all[var_name][t]
            if len(all_gt_frames[var_name]) == t:  # Only append once
                all_gt_frames[var_name].append(gt_var)
        
        # Clean up GPU memory
        del sparse_resized, interp_resized
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Generate outputs for each variable and each iteration count
    for var_name in variables:
        print(f"\nGenerating outputs for variable: {var_name}")
        
        for n_iters in denoise_iterations:
            print(f"  Processing {n_iters} iteration(s)...")
            
            # Create GIF with comparison frames
            gif_frames = []
            for t in range(num_timesteps):
                frame_img = create_comparison_frame(
                    all_gt_frames[var_name][t],
                    all_sparse_frames[var_name][n_iters][t],
                    all_interp_frames[var_name][n_iters][t],
                    var_name,
                    probe_coords,
                )
                gif_frames.append(frame_img)
            
            gif_path = output_dir / f"{var_name}_reconstruction_iter{n_iters}.gif"
            imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
            print(f"    Saved GIF: {gif_path.name}")
            
            # Create single summary image (last frame)
            summary_img = create_comparison_frame(
                all_gt_frames[var_name][-1],
                all_sparse_frames[var_name][n_iters][-1],
                all_interp_frames[var_name][n_iters][-1],
                var_name,
                probe_coords,
            )
            summary_path = output_dir / f"{var_name}_summary_iter{n_iters}.png"
            imageio.imwrite(summary_path, summary_img)
            print(f"    Saved summary: {summary_path.name}")
    
    print(f"\nInference complete! Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="VAE inference with probe data")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/bout_data",
        help="Directory with ground truth .npy files"
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/probe_data",
        help="Directory with probe data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/results/vae_inference",
        help="Output directory"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "te", "ti", "phi"],
        help="Variables to process"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Model input size"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Noise std for sparse probe images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--denoise-iterations",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Number of denoising iterations to test"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model (auto-detects input size from checkpoint)
    model, model_input_size = load_model(
        Path(args.checkpoint),
        device,
        num_vars=len(args.variables)
    )
    
    print(f"\nModel was trained on: {model_input_size}×{model_input_size}")
    print(f"Inference will use resize: {args.resize}×{args.resize}")
    
    # Run inference
    run_inference(
        model=model,
        data_dir=Path(args.data_dir),
        probe_dir=Path(args.probe_dir),
        output_dir=Path(args.output),
        variables=args.variables,
        device=device,
        resize=args.resize,  # Use user-specified resize for inference
        noise_std=args.noise_std,
        denoise_iterations=args.denoise_iterations,
    )


if __name__ == "__main__":
    main()
