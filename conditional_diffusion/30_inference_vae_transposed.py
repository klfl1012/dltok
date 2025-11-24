#!/usr/bin/env python3
"""
Inference script for VAE with transposed convolutions.
Uses the same probe-based reconstruction approach as the original VAE.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import importlib.util

current_dir = Path(__file__).parent

# Load model
spec = importlib.util.spec_from_file_location("model_27", current_dir / "27_model_vae_transposed.py")
model_27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_27)
VAETransposed = model_27.VAETransposed


def load_model(checkpoint_path: Path, device, num_vars: int = 4):
    """Load trained VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer input size from decoder.fc weight shape
    fc_out_features = checkpoint['model_state_dict']['decoder.fc.weight'].shape[0]
    spatial_size = int(np.sqrt(fc_out_features / 256))
    input_size = spatial_size * 16
    
    print(f"Detected model input size: {input_size}×{input_size}")
    
    model = VAETransposed(
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
    """Create sparse probe image with noise background."""
    H, W = target_size
    image = np.random.randn(H, W) * noise_std
    
    # Scale probe coordinates to target size
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
    target_size: tuple = (1024, 1024),
) -> np.ndarray:
    """Create probe image with bilinear interpolation."""
    from scipy.interpolate import interp1d
    
    H, W = target_size
    
    # Scale probe coordinates
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


def create_comparison_frame(
    ground_truth: np.ndarray,
    sparse_recon: np.ndarray,
    interp_recon: np.ndarray,
    variable_name: str,
    probe_coords: np.ndarray = None,
    cmap: str = 'viridis',
) -> np.ndarray:
    """Create comparison frame with ground truth and two reconstructions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
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
    
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sparse_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'Sparse Error (MAE={sparse_error.mean():.4f})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(interp_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 2].set_title(f'Interp Error (MAE={interp_error.mean():.4f})')
    axes[1, 2].axis('off')
    
    plt.colorbar(im, ax=axes[0, :], fraction=0.046, pad=0.04)
    
    if probe_coords is not None:
        for ax in axes[0, :]:
            ax.scatter(probe_coords[:, 0], probe_coords[:, 1], 
                      c='red', s=10, marker='x', alpha=0.5)
    
    plt.tight_layout()
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]
    
    plt.close(fig)
    
    return image


@torch.no_grad()
def run_inference(
    model,
    data_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    variables: list,
    device,
    resize: int = 256,
    noise_std: float = 0.1,
    denoise_iterations: list = [1, 2, 5, 10],
):
    """Run inference on probe data and generate visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running inference...")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load probe coordinates
    probe_coords = np.load(probe_dir / "probe_coordinates.npy")
    print(f"Probe coordinates: {probe_coords.shape}")
    
    # Load data and compute stats
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
    
    num_timesteps = gt_data_all[variables[0]].shape[0]
    print(f"Number of timesteps: {num_timesteps}")
    print(f"Denoising iterations to test: {denoise_iterations}")
    
    # Storage for frames
    all_gt_frames = {var: [] for var in variables}
    all_sparse_frames = {var: {n: [] for n in denoise_iterations} for var in variables}
    all_interp_frames = {var: {n: [] for n in denoise_iterations} for var in variables}
    
    # Process all timesteps
    for t in range(num_timesteps):
        if t % 10 == 0:
            print(f"  Processing frame {t}/{num_timesteps}")
        
        # Prepare multi-channel inputs
        sparse_channels = []
        interp_channels = []
        gt_channels = []
        
        for var_name in variables:
            gt_frame = gt_data_all[var_name][t]
            gt_normalized = (gt_frame - stats[var_name]['mean']) / (stats[var_name]['std'] + 1e-8)
            gt_channels.append(gt_normalized)
            
            probe_vals = probe_data_all[var_name][t]
            probe_vals_norm = (probe_vals - stats[var_name]['mean']) / (stats[var_name]['std'] + 1e-8)
            
            sparse_img = create_probe_image_sparse(probe_coords, probe_vals_norm, (1024, 1024), noise_std)
            interp_img = create_probe_image_interpolated(probe_coords, probe_vals_norm, (1024, 1024))
            
            sparse_channels.append(sparse_img)
            interp_channels.append(interp_img)
        
        # Stack and resize
        sparse_multi = np.stack(sparse_channels, axis=0)
        interp_multi = np.stack(interp_channels, axis=0)
        
        sparse_tensor = torch.from_numpy(sparse_multi).float().unsqueeze(0)
        interp_tensor = torch.from_numpy(interp_multi).float().unsqueeze(0)
        
        sparse_resized = F.interpolate(sparse_tensor, size=(resize, resize), mode='bilinear', align_corners=False)
        interp_resized = F.interpolate(interp_tensor, size=(resize, resize), mode='bilinear', align_corners=False)
        
        sparse_current = sparse_resized.clone().to(device)
        interp_current = interp_resized.clone().to(device)
        
        sparse_probe_input = sparse_resized.to(device)
        interp_probe_input = interp_resized.to(device)
        
        # Iterative denoising
        for n_iters in denoise_iterations:
            sparse_denoised = sparse_current.clone()
            interp_denoised = interp_current.clone()
            
            for _ in range(n_iters):
                sparse_recon, _, _ = model(sparse_denoised)
                interp_recon, _, _ = model(interp_denoised)
                
                # Ensure reconstruction is same size as input (model might output at different resolution)
                if sparse_recon.shape != sparse_denoised.shape:
                    sparse_recon = F.interpolate(sparse_recon, size=sparse_denoised.shape[-2:], mode='bilinear', align_corners=False)
                    interp_recon = F.interpolate(interp_recon, size=interp_denoised.shape[-2:], mode='bilinear', align_corners=False)
                
                # Preserve probe values
                sparse_denoised = sparse_recon * 0.7 + sparse_probe_input * 0.3
                interp_denoised = interp_recon * 0.7 + interp_probe_input * 0.3
            
            # Upsample back to original size
            sparse_final = F.interpolate(sparse_denoised, size=(1024, 1024), mode='bilinear', align_corners=False)
            interp_final = F.interpolate(interp_denoised, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            sparse_np = sparse_final.squeeze(0).cpu().numpy()
            interp_np = interp_final.squeeze(0).cpu().numpy()
            
            # Store per variable
            for var_idx, var_name in enumerate(variables):
                mean = stats[var_name]['mean']
                std = stats[var_name]['std']
                
                sparse_denorm = sparse_np[var_idx] * std + mean
                interp_denorm = interp_np[var_idx] * std + mean
                
                all_sparse_frames[var_name][n_iters].append(sparse_denorm)
                all_interp_frames[var_name][n_iters].append(interp_denorm)
        
        # Store ground truth
        for var_name in variables:
            gt_frame = gt_data_all[var_name][t]
            all_gt_frames[var_name].append(gt_frame)
        
        del sparse_current, interp_current, sparse_probe_input, interp_probe_input
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Generate outputs
    for var_name in variables:
        print(f"\nGenerating outputs for variable: {var_name}")
        
        for n_iters in denoise_iterations:
            gif_frames = []
            
            for t in range(num_timesteps):
                frame_img = create_comparison_frame(
                    all_gt_frames[var_name][t],
                    all_sparse_frames[var_name][n_iters][t],
                    all_interp_frames[var_name][n_iters][t],
                    f"{var_name} (iter={n_iters})",
                    probe_coords,
                )
                gif_frames.append(frame_img)
            
            gif_path = output_dir / f"{var_name}_transposed_iter{n_iters}.gif"
            imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
            print(f"  Saved: {gif_path.name}")
    
    print(f"\nInference complete! Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="VAE Transposed inference with probe data")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--probe-dir", type=str, default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/results/vae_transposed_inference")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--denoise-iterations", type=int, nargs="+", default=[1, 2, 5, 10])
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    model, model_input_size = load_model(
        Path(args.checkpoint),
        device,
        num_vars=len(args.variables)
    )
    
    print(f"\nModel was trained on: {model_input_size}×{model_input_size}")
    print(f"Inference will use resize: {args.resize}×{args.resize}")
    
    run_inference(
        model=model,
        data_dir=Path(args.data_dir),
        probe_dir=Path(args.probe_dir),
        output_dir=Path(args.output),
        variables=args.variables,
        device=device,
        resize=args.resize,
        noise_std=args.noise_std,
        denoise_iterations=args.denoise_iterations,
    )


if __name__ == "__main__":
    main()
