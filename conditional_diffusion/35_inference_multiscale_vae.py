#!/usr/bin/env python3
"""
Inference script for Multi-Scale VAE with Elbow Loss.
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

current_dir = Path(__file__).parent

# Load dataset
spec = importlib.util.spec_from_file_location("dataset_4", current_dir / "4_dataset.py")
dataset_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_4)
build_dataloaders = dataset_4.build_dataloaders

# Load model
spec = importlib.util.spec_from_file_location("model_vae", current_dir / "33_model_multiscale_vae_elbow.py")
model_vae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_vae)
MultiScaleVAE = model_vae.MultiScaleVAE

def compute_metrics(model, dataloader, device, output_dir):
    """Compute comprehensive metrics for all samples."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    metrics_per_scale = {}
    mse_values = {}
    
    with torch.no_grad():
        all_data = []
        all_outputs = {}
        
        # Collect all samples
        for data in dataloader:
            data = data.to(device)
            outputs, mu, logvar = model(data)
            
            all_data.append(data)
            for scale in outputs.keys():
                if scale not in all_outputs:
                    all_outputs[scale] = []
                all_outputs[scale].append(outputs[scale])
        
        # Concatenate all batches
        all_data = torch.cat(all_data, dim=0)
        for scale in all_outputs.keys():
            all_outputs[scale] = torch.cat(all_outputs[scale], dim=0)
        
        # Compute metrics for each scale
        for scale, output in all_outputs.items():
            print(f"Computing metrics for {scale}x{scale}...")
            
            # Downsample target if needed
            if all_data.shape[-1] != scale:
                target_scaled = torch.nn.functional.interpolate(
                    all_data, size=(scale, scale), mode='bilinear', align_corners=False
                )
            else:
                target_scaled = all_data
            
            # Normalize to [0, 1] for metrics
            output_norm = (output - output.min()) / (output.max() - output.min() + 1e-8)
            target_norm = (target_scaled - target_scaled.min()) / (target_scaled.max() - target_scaled.min() + 1e-8)
            
            # MSE
            mse = torch.nn.functional.mse_loss(output, target_scaled).item()
            
            # SSIM (requires at least 2 channels or grayscale)
            ssim_val = ssim_metric(output_norm, target_norm).item()
            
            # MS-SSIM (requires minimum size)
            if scale >= 160:  # MS-SSIM needs at least 160x160
                ms_ssim_val = ms_ssim_metric(output_norm, target_norm).item()
            else:
                ms_ssim_val = None
            
            # LPIPS (requires 3 channels and [0, 1] range)
            # Repeat channels to make it 3-channel if needed
            if output_norm.shape[1] == 2:
                # Duplicate first channel to make 3 channels (using normalized versions)
                output_3ch = torch.cat([output_norm, output_norm[:, :1, :, :]], dim=1)
                target_3ch = torch.cat([target_norm, target_norm[:, :1, :, :]], dim=1)
            else:
                output_3ch = output_norm.repeat(1, 3, 1, 1)
                target_3ch = target_norm.repeat(1, 3, 1, 1)
            
            lpips_val = lpips_metric(output_3ch, target_3ch).item()
            
            metrics_per_scale[scale] = {
                'mse': mse,
                'ssim': ssim_val,
                'ms_ssim': ms_ssim_val,
                'lpips': lpips_val,
            }
    
    # Save metrics to file
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("VAE Reconstruction Metrics\\n")
        f.write("=" * 60 + "\\n\\n")
        
        for scale in sorted(metrics_per_scale.keys()):
            f.write(f"Scale: {scale}x{scale}\\n")
            f.write(f"  MSE:     {metrics_per_scale[scale]['mse']:.6f}\\n")
            f.write(f"  SSIM:    {metrics_per_scale[scale]['ssim']:.6f}\\n")
            if metrics_per_scale[scale]['ms_ssim'] is not None:
                f.write(f"  MS-SSIM: {metrics_per_scale[scale]['ms_ssim']:.6f}\\n")
            f.write(f"  LPIPS:   {metrics_per_scale[scale]['lpips']:.6f}\\n")
            f.write("\\n")
    
    print(f"\\nMetrics saved to {metrics_file}")
    return metrics_per_scale

def visualize_reconstructions(model, dataloader, device, output_dir, num_samples=5):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch of data
    data = next(iter(dataloader))
    data = data.to(device)
    
    with torch.no_grad():
        outputs, mu, logvar = model(data)
    
    # outputs is a dict {64: tensor, 128: tensor, 256: tensor}
    scales = sorted(outputs.keys())
    num_cols = len(scales) + 1
    
    # Plotting
    for i in range(min(num_samples, data.size(0))):
        fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))
        
        # Original Input (256x256)
        # Channel 0 (n)
        im0 = axes[0, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
        axes[0, 0].set_title("Original (n)")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Channel 1 (phi)
        im1 = axes[1, 0].imshow(data[i, 1].cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title("Original (phi)")
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Reconstructions at different scales
        scales = sorted(outputs.keys())
        for idx, scale in enumerate(scales):
            recon = outputs[scale][i] # (C, H, W)
            
            # Channel 0 (n)
            im_n = axes[0, idx+1].imshow(recon[0].cpu().numpy(), cmap='viridis')
            axes[0, idx+1].set_title(f"Recon {scale}x{scale} (n)")
            plt.colorbar(im_n, ax=axes[0, idx+1])
            
            # Channel 1 (phi)
            im_p = axes[1, idx+1].imshow(recon[1].cpu().numpy(), cmap='viridis')
            axes[1, idx+1].set_title(f"Recon {scale}x{scale} (phi)")
            plt.colorbar(im_p, ax=axes[1, idx+1])
            
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i}.png")
        plt.close()
        
    print(f"Saved {num_samples} visualization samples to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/tcv_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "phi"])
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/results/multiscale_vae_elbow")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimension (auto-detect if not specified)")
    parser.add_argument("--base-channels", type=int, default=None, help="Override base channels (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint to get configuration
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Try to auto-detect latent_dim and base_channels from checkpoint
    if args.latent_dim is None:
        # Check if we can infer from state_dict
        if 'encoder.fc_mu.bias' in checkpoint['model_state_dict']:
            latent_dim = checkpoint['model_state_dict']['encoder.fc_mu.bias'].shape[0]
            print(f"Auto-detected latent_dim: {latent_dim}")
        else:
            latent_dim = 256  # fallback
            print(f"Using default latent_dim: {latent_dim}")
    else:
        latent_dim = args.latent_dim
        print(f"Using specified latent_dim: {latent_dim}")
    
    if args.base_channels is None:
        # Check if we can infer from state_dict
        if 'encoder.conv1.0.weight' in checkpoint['model_state_dict']:
            base_channels = checkpoint['model_state_dict']['encoder.conv1.0.weight'].shape[0]
            print(f"Auto-detected base_channels: {base_channels}")
        else:
            base_channels = 32  # fallback
            print(f"Using default base_channels: {base_channels}")
    else:
        base_channels = args.base_channels
        print(f"Using specified base_channels: {base_channels}")
    
    # Load Model
    model = MultiScaleVAE(
        in_channels=len(args.variables),
        latent_dim=latent_dim,
        base_channels=base_channels,
        target_sizes=[64, 128, 256, 512],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (Epoch {checkpoint['epoch']})")
    
    # Load Data
    _, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=8,
        num_workers=4,
    )
    
    # Compute metrics
    print("\\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    metrics = compute_metrics(model, val_loader, device, args.output)
    
    # Visualize reconstructions
    print("\\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    visualize_reconstructions(model, val_loader, device, args.output)

if __name__ == "__main__":
    main()
