#!/usr/bin/env python3
"""
Inference script for Progressive Multi-Scale VAE.

Tests reconstruction quality at multiple resolutions and generates
comparison visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from matplotlib import cm

# Import model
import importlib.util

current_dir = Path(__file__).parent

# Load progressive VAE model
spec = importlib.util.spec_from_file_location("model_progressive", current_dir / "18_model_progressive_vae.py")
model_progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_progressive)
ProgressiveVAE = model_progressive.ProgressiveVAE


def load_model(checkpoint_path: Path, device, num_vars: int = 4):
    """Load trained Progressive VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get target sizes from checkpoint
    target_sizes = checkpoint.get('target_sizes', [64, 128, 256])
    
    print(f"Detected progressive scales: {target_sizes}")
    
    model = ProgressiveVAE(
        in_channels=num_vars,
        latent_dim=256,
        base_channels=32,
        target_sizes=target_sizes,
        intermediate_noise_scale=0.0,  # No noise during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model, target_sizes


def create_probe_image_sparse(
    probe_coords: np.ndarray,
    probe_values: np.ndarray,
    target_size: tuple = (256, 256),
    noise_std: float = 0.1,
) -> np.ndarray:
    """Create sparse probe image with noise background."""
    H, W = target_size
    image = np.random.randn(H, W) * noise_std
    
    # Scale probe coordinates from 1024x1024 to target_size
    scale_x = W / 1024.0
    scale_z = H / 1024.0
    
    for (x, z), value in zip(probe_coords, probe_values):
        x_scaled = int(x * scale_x)
        z_scaled = int(z * scale_z)
        if 0 <= x_scaled < W and 0 <= z_scaled < H:
            image[z_scaled, x_scaled] = value
    
    return image


def create_probe_image_interpolated(
    probe_coords: np.ndarray,
    probe_values: np.ndarray,
    target_size: tuple = (256, 256),
) -> np.ndarray:
    """Create probe image with bilinear interpolation."""
    from scipy.interpolate import interp1d
    
    H, W = target_size
    
    # Scale probe coordinates from 1024x1024 to target_size
    scale_x = W / 1024.0
    scale_z = H / 1024.0
    probe_coords_scaled = probe_coords.copy().astype(float)
    probe_coords_scaled[:, 0] *= scale_x
    probe_coords_scaled[:, 1] *= scale_z
    
    unique_z = np.unique(probe_coords_scaled[:, 1])
    
    if len(unique_z) == 1:
        # Horizontal line interpolation
        probe_z = int(unique_z[0])
        probe_x = probe_coords_scaled[:, 0]
        
        sort_idx = np.argsort(probe_x)
        probe_x_sorted = probe_x[sort_idx]
        probe_vals_sorted = probe_values[sort_idx]
        
        interpolator = interp1d(
            probe_x_sorted,
            probe_vals_sorted,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        x_coords = np.arange(W)
        interpolated_line = interpolator(x_coords)
        
        image = np.zeros((H, W))
        image[probe_z, :] = interpolated_line
        
        # Extend vertically with decay
        for z in range(H):
            distance = abs(z - probe_z)
            decay = np.exp(-distance / (H * 0.1))
            image[z, :] = interpolated_line * decay
        
        return image
    else:
        # 2D interpolation
        from scipy.interpolate import griddata
        
        grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(H))
        
        image = griddata(
            probe_coords_scaled,
            probe_values,
            (grid_x, grid_z),
            method='linear',
            fill_value=0.0
        )
        
        return image


def create_multiscale_comparison(
    ground_truth: np.ndarray,
    multiscale_outputs: dict,
    probe_input: np.ndarray,
    variable_name: str,
    probe_coords: np.ndarray = None,
    cmap: str = 'viridis',
) -> np.ndarray:
    """
    Create comparison showing all scales + ground truth.
    
    Returns RGB image as numpy array.
    """
    num_scales = len(multiscale_outputs)
    fig, axes = plt.subplots(2, num_scales + 1, figsize=(5 * (num_scales + 1), 10))
    
    # Compute global vmin/vmax
    all_data = [ground_truth, probe_input] + list(multiscale_outputs.values())
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)
    
    # Row 1: Reconstructions at each scale + ground truth
    # First column: probe input
    axes[0, 0].imshow(probe_input, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{variable_name} - Probe Input')
    axes[0, 0].axis('off')
    
    if probe_coords is not None:
        axes[0, 0].scatter(probe_coords[:, 0], probe_coords[:, 1], 
                          c='red', s=10, alpha=0.5, marker='x')
    
    # Other columns: progressive reconstructions
    for idx, (size, output) in enumerate(sorted(multiscale_outputs.items())):
        im = axes[0, idx + 1].imshow(output, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, idx + 1].set_title(f'{variable_name} - {size}×{size}')
        axes[0, idx + 1].axis('off')
    
    # Row 2: Errors vs ground truth
    axes[1, 0].imshow(ground_truth, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Compute errors for each scale
    error_max = 0.0
    errors = {}
    for size, output in multiscale_outputs.items():
        # Resize ground truth to match scale
        gt_scaled = F.interpolate(
            torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).float(),
            size=(size, size),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        error = np.abs(gt_scaled - output)
        errors[size] = error
        error_max = max(error_max, error.max())
    
    for idx, (size, error) in enumerate(sorted(errors.items())):
        mae = error.mean()
        axes[1, idx + 1].imshow(error, cmap='hot', vmin=0, vmax=error_max)
        axes[1, idx + 1].set_title(f'{size}×{size} Error (MAE={mae:.4f})')
        axes[1, idx + 1].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[0, :], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]
    
    plt.close(fig)
    
    return image


@torch.no_grad()
def run_inference(
    model: ProgressiveVAE,
    data_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    variables: list,
    device,
    target_sizes: list,
    noise_std: float = 0.1,
    max_resolution: int = 256,
    num_timesteps: int = None,
):
    """
    Run inference and generate multi-scale visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Progressive VAE inference...")
    print(f"Device: {device}")
    print(f"Progressive scales: {target_sizes}")
    print(f"Output: {output_dir}")
    
    # Load probe coordinates
    probe_coords = np.load(probe_dir / "probe_coordinates.npy")
    print(f"Probe coordinates: {probe_coords.shape}")
    
    # Load data and compute stats
    stats = {}
    gt_data_all = {}
    probe_data_all = {}
    
    for var_name in variables:
        # Ground truth
        gt_path = data_dir / f"{var_name}.npy"
        gt_data = np.load(gt_path)
        
        # Probe data
        probe_path = probe_dir / f"probe_{var_name}.npy"
        probe_data = np.load(probe_path)
        
        # Compute stats
        mean = gt_data.mean()
        std = gt_data.std()
        stats[var_name] = {'mean': mean, 'std': std}
        
        # Normalize
        gt_data = (gt_data - mean) / (std + 1e-8)
        probe_data = (probe_data - mean) / (std + 1e-8)
        
        gt_data_all[var_name] = gt_data
        probe_data_all[var_name] = probe_data
        
        print(f"  {var_name}: GT {gt_data.shape}, Probe {probe_data.shape}")
    
    # Determine timesteps to process
    total_timesteps = gt_data_all[variables[0]].shape[0]
    if num_timesteps is None or num_timesteps > total_timesteps:
        num_timesteps = total_timesteps
    
    print(f"Processing {num_timesteps} timesteps")
    
    # Storage for GIFs (one per variable per scale)
    all_frames = {var: {size: [] for size in target_sizes} for var in variables}
    all_comparison_frames = {var: [] for var in variables}
    
    # Process each timestep
    for t in range(num_timesteps):
        if t % 10 == 0:
            print(f"Processing timestep {t}/{num_timesteps}...")
        
        # Prepare input: stack all variables
        input_channels = []
        gt_channels = []
        
        for var_name in variables:
            # Create probe input (interpolated)
            probe_vals = probe_data_all[var_name][t]
            probe_input = create_probe_image_interpolated(
                probe_coords, probe_vals, (max_resolution, max_resolution)
            )
            input_channels.append(probe_input)
            
            # Ground truth (resized to max resolution)
            gt_full = gt_data_all[var_name][t]
            gt_resized = F.interpolate(
                torch.from_numpy(gt_full).unsqueeze(0).unsqueeze(0).float(),
                size=(max_resolution, max_resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            gt_channels.append(gt_resized)
        
        # Stack to multi-channel tensor
        input_tensor = np.stack(input_channels, axis=0)  # (C, H, W)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)
        
        # Forward pass -> multi-scale outputs
        outputs, _, _ = model(input_tensor)
        
        # Process each variable
        for var_idx, var_name in enumerate(variables):
            probe_input = input_channels[var_idx]
            gt = gt_channels[var_idx]
            
            # Extract outputs for this variable at each scale
            var_outputs = {}
            for size in target_sizes:
                output = outputs[size][0, var_idx].cpu().numpy()  # (H, W)
                var_outputs[size] = output
                
                # Store for GIF
                all_frames[var_name][size].append(output)
            
            # Create multi-scale comparison
            comparison = create_multiscale_comparison(
                gt, var_outputs, probe_input, var_name, probe_coords
            )
            all_comparison_frames[var_name].append(comparison)
    
    # Save GIFs
    print("\nSaving GIFs...")
    for var_name in variables:
        # Multi-scale comparison GIF
        comp_gif_path = output_dir / f"{var_name}_multiscale_comparison.gif"
        imageio.mimsave(comp_gif_path, all_comparison_frames[var_name], fps=5)
        print(f"  Saved {comp_gif_path}")
        
        # Individual scale GIFs
        for size in target_sizes:
            frames_normalized = []
            frames_array = np.array(all_frames[var_name][size])
            vmin, vmax = frames_array.min(), frames_array.max()
            
            for frame in all_frames[var_name][size]:
                # Normalize and convert to RGB
                frame_norm = (frame - vmin) / (vmax - vmin + 1e-8)
                frame_rgb = (cm.viridis(frame_norm)[:, :, :3] * 255).astype(np.uint8)
                frames_normalized.append(frame_rgb)
            
            gif_path = output_dir / f"{var_name}_scale_{size}.gif"
            imageio.mimsave(gif_path, frames_normalized, fps=5)
            print(f"  Saved {gif_path}")
    
    print(f"\nInference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Progressive VAE inference")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, 
                       default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--probe-dir", type=str,
                       default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--output", type=str,
                       default="/dtu/blackhole/1b/223803/results/progressive_vae_inference")
    parser.add_argument("--variables", type=str, nargs="+",
                       default=["n", "te", "ti", "phi"])
    parser.add_argument("--max-resolution", type=int, default=256,
                       help="Max resolution for input/comparison")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-timesteps", type=int, default=None,
                       help="Number of timesteps to process (None = all)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    model, target_sizes = load_model(
        Path(args.checkpoint),
        device,
        num_vars=len(args.variables)
    )
    
    # Run inference
    run_inference(
        model=model,
        data_dir=Path(args.data_dir),
        probe_dir=Path(args.probe_dir),
        output_dir=Path(args.output),
        variables=args.variables,
        device=device,
        target_sizes=target_sizes,
        noise_std=args.noise_std,
        max_resolution=args.max_resolution,
        num_timesteps=args.num_timesteps,
    )


if __name__ == "__main__":
    main()
