#!/usr/bin/env python3
"""
Inference script for Temporal Progressive VAE.

Processes probe sequences (10 timesteps) and reconstructs current frame
at multiple resolutions.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import importlib.util

current_dir = Path(__file__).parent

# Load progressive VAE model
spec = importlib.util.spec_from_file_location("model_progressive", current_dir / "18_model_progressive_vae.py")
model_progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_progressive)
TemporalProgressiveVAE = model_progressive.TemporalProgressiveVAE


def load_model(checkpoint_path: Path, device, num_vars: int = 4):
    """Load trained Temporal Progressive VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    target_sizes = checkpoint.get('target_sizes', [64, 128, 256])
    sequence_length = checkpoint.get('sequence_length', 10)
    
    print(f"Detected progressive scales: {target_sizes}")
    print(f"Sequence length: {sequence_length}")
    
    model = TemporalProgressiveVAE(
        in_channels=num_vars,
        latent_dim=256,
        base_channels=32,
        target_sizes=target_sizes,
        hidden_dims=[64, 128],
        sequence_length=sequence_length,
        intermediate_noise_scale=0.0,  # No noise during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model, target_sizes, sequence_length


def create_probe_image_interpolated(
    probe_coords: np.ndarray,
    probe_values: np.ndarray,
    target_size: tuple = (256, 256),
) -> np.ndarray:
    """Create probe image with RBF interpolation (robust to colinear points)."""
    from scipy.interpolate import Rbf, interp1d
    
    H, W = target_size
    
    # Scale probe coordinates from 1024x1024 to target_size
    scale_x = W / 1024.0
    scale_z = H / 1024.0
    probe_coords_scaled = probe_coords.copy().astype(float)
    probe_coords_scaled[:, 0] *= scale_x
    probe_coords_scaled[:, 1] *= scale_z
    
    # Check if points are colinear
    unique_x = np.unique(probe_coords_scaled[:, 0])
    unique_z = np.unique(probe_coords_scaled[:, 1])
    
    if len(unique_x) == 1 or len(unique_z) == 1:
        # Fallback to 1D interpolation
        if len(unique_z) == 1:
            # Horizontal line
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
            if 0 <= probe_z < H:
                image[probe_z, :] = interpolated_line
                for z in range(H):
                    distance = abs(z - probe_z)
                    decay = np.exp(-distance / (H * 0.1))
                    image[z, :] = interpolated_line * decay
        else:
            # Vertical line
            probe_x = int(unique_x[0])
            probe_z = probe_coords_scaled[:, 1]
            
            sort_idx = np.argsort(probe_z)
            probe_z_sorted = probe_z[sort_idx]
            probe_vals_sorted = probe_values[sort_idx]
            
            interpolator = interp1d(
                probe_z_sorted,
                probe_vals_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            z_coords = np.arange(H)
            interpolated_line = interpolator(z_coords)
            
            image = np.zeros((H, W))
            if 0 <= probe_x < W:
                image[:, probe_x] = interpolated_line
                for x in range(W):
                    distance = abs(x - probe_x)
                    decay = np.exp(-distance / (W * 0.1))
                    image[:, x] = interpolated_line * decay
    else:
        # 2D RBF interpolation
        try:
            rbf = Rbf(
                probe_coords_scaled[:, 0],
                probe_coords_scaled[:, 1],
                probe_values,
                function='linear',
                smooth=0.1
            )
            
            grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(H))
            image = rbf(grid_x, grid_z)
        except Exception:
            # Fallback: sparse representation with nearest-neighbor fill
            image = np.zeros((H, W))
            for (x, z), value in zip(probe_coords_scaled.astype(int), probe_values):
                if 0 <= x < W and 0 <= z < H:
                    image[int(z), int(x)] = value
    
    return image


def create_multiscale_comparison(
    ground_truth: np.ndarray,
    multiscale_outputs: dict,
    probe_sequence_last: np.ndarray,
    variable_name: str,
    cmap: str = 'viridis',
) -> np.ndarray:
    """
    Create comparison showing all scales + ground truth.
    
    Returns RGB image as numpy array.
    """
    num_scales = len(multiscale_outputs)
    fig, axes = plt.subplots(2, num_scales + 1, figsize=(5 * (num_scales + 1), 10))
    
    # Compute global vmin/vmax
    all_data = [ground_truth, probe_sequence_last] + list(multiscale_outputs.values())
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)
    
    # Row 1: Reconstructions at each scale + ground truth
    # First column: last probe input
    axes[0, 0].imshow(probe_sequence_last, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{variable_name} - Last Probe Input (t-1)')
    axes[0, 0].axis('off')
    
    # Other columns: progressive reconstructions
    for idx, (size, output) in enumerate(sorted(multiscale_outputs.items())):
        im = axes[0, idx + 1].imshow(output, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, idx + 1].set_title(f'{variable_name} - {size}×{size}')
        axes[0, idx + 1].axis('off')
    
    # Row 2: Errors vs ground truth
    axes[1, 0].imshow(ground_truth, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth (t)')
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
    model: TemporalProgressiveVAE,
    data_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    variables: list,
    device,
    target_sizes: list,
    sequence_length: int = 10,
    max_resolution: int = 256,
    num_samples: int = None,
):
    """
    Run inference and generate multi-scale visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Temporal Progressive VAE inference...")
    print(f"Device: {device}")
    print(f"Progressive scales: {target_sizes}")
    print(f"Sequence length: {sequence_length}")
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
    
    # Determine valid timesteps (need sequence_length history)
    total_timesteps = gt_data_all[variables[0]].shape[0]
    valid_timesteps = np.arange(sequence_length, total_timesteps)
    
    if num_samples is not None and num_samples < len(valid_timesteps):
        # Sample evenly
        step = max(1, len(valid_timesteps) // num_samples)
        valid_timesteps = valid_timesteps[::step][:num_samples]
    
    print(f"Processing {len(valid_timesteps)} samples")
    
    # Storage for GIFs (one per variable per scale)
    all_frames = {var: {size: [] for size in target_sizes} for var in variables}
    all_comparison_frames = {var: [] for var in variables}
    
    # Storage for metrics
    all_maes = {var: {size: [] for size in target_sizes} for var in variables}
    
    # Process each timestep
    for sample_idx, t_current in enumerate(valid_timesteps):
        if sample_idx % 10 == 0:
            print(f"Processing sample {sample_idx}/{len(valid_timesteps)} (t={t_current})...")
        
        t_start = t_current - sequence_length
        
        # Build probe sequence for all variables
        probe_sequence = []
        for t in range(t_start, t_current):
            frame_probes = []
            for var_name in variables:
                probe_vals = probe_data_all[var_name][t]
                probe_img = create_probe_image_interpolated(
                    probe_coords, probe_vals, (max_resolution, max_resolution)
                )
                frame_probes.append(probe_img)
            
            probe_sequence.append(np.stack(frame_probes, axis=0))  # (C, H, W)
        
        probe_sequence = np.stack(probe_sequence, axis=0)  # (T, C, H, W)
        probe_sequence_tensor = torch.from_numpy(probe_sequence).unsqueeze(0).float().to(device)  # (1, T, C, H, W)
        
        # Forward pass -> multi-scale outputs
        outputs, _, _ = model(probe_sequence_tensor)
        
        # Process each variable
        for var_idx, var_name in enumerate(variables):
            # Ground truth (current timestep)
            gt = gt_data_all[var_name][t_current]
            gt_resized = F.interpolate(
                torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float(),
                size=(max_resolution, max_resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Last probe input
            probe_last = probe_sequence[-1, var_idx]  # (H, W)
            
            # Multi-scale outputs
            multiscale_outputs = {}
            for size in target_sizes:
                output = outputs[size][0, var_idx].cpu().numpy()  # (H, W)
                multiscale_outputs[size] = output
                
                # Compute MAE
                gt_scaled = F.interpolate(
                    torch.from_numpy(gt_resized).unsqueeze(0).unsqueeze(0).float(),
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                mae = np.abs(gt_scaled - output).mean()
                all_maes[var_name][size].append(mae)
                
                # Store frame for scale-specific GIF
                # Denormalize for visualization
                mean = stats[var_name]['mean']
                std = stats[var_name]['std']
                output_denorm = output * std + mean
                gt_scaled_denorm = gt_scaled * std + mean
                
                # Create simple comparison
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                vmin = min(gt_scaled_denorm.min(), output_denorm.min())
                vmax = max(gt_scaled_denorm.max(), output_denorm.max())
                
                axes[0].imshow(gt_scaled_denorm, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0].set_title(f'GT {var_name} (t={t_current})')
                axes[0].axis('off')
                
                axes[1].imshow(output_denorm, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title(f'Pred {size}×{size}')
                axes[1].axis('off')
                
                error = np.abs(gt_scaled_denorm - output_denorm)
                axes[2].imshow(error, cmap='hot')
                axes[2].set_title(f'Error (MAE={mae:.4f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
                plt.close(fig)
                
                all_frames[var_name][size].append(frame)
            
            # Create full comparison frame
            comp_frame = create_multiscale_comparison(
                gt_resized, multiscale_outputs, probe_last, var_name
            )
            all_comparison_frames[var_name].append(comp_frame)
    
    # Save GIFs
    print("\nSaving GIFs...")
    for var_name in variables:
        # Multi-scale comparison GIF
        comp_gif_path = output_dir / f"{var_name}_temporal_multiscale_comparison.gif"
        imageio.mimsave(comp_gif_path, all_comparison_frames[var_name], fps=5)
        print(f"  Saved {comp_gif_path}")
        
        # Individual scale GIFs
        for size in target_sizes:
            scale_gif_path = output_dir / f"{var_name}_temporal_{size}x{size}.gif"
            imageio.mimsave(scale_gif_path, all_frames[var_name][size], fps=5)
            print(f"  Saved {scale_gif_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Mean MAE across all samples)")
    print("="*70)
    for var_name in variables:
        print(f"\n{var_name}:")
        for size in target_sizes:
            mean_mae = np.mean(all_maes[var_name][size])
            std_mae = np.std(all_maes[var_name][size])
            print(f"  {size}×{size}: MAE = {mean_mae:.4f} ± {std_mae:.4f}")
    
    print(f"\nInference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Temporal Progressive VAE inference")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, 
                       default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--probe-dir", type=str,
                       default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--output", type=str,
                       default="/dtu/blackhole/1b/223803/results/temporal_progressive_vae_inference")
    parser.add_argument("--variables", type=str, nargs="+",
                       default=["n", "te", "ti", "phi"])
    parser.add_argument("--max-resolution", type=int, default=256,
                       help="Max resolution for input/comparison")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to process (None = all)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    model, target_sizes, sequence_length = load_model(
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
        sequence_length=sequence_length,
        max_resolution=args.max_resolution,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
