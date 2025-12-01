#!/usr/bin/env python3
"""
Inference script for Probe-based Reconstruction.

Visualizes reconstructions from probe measurements using the trained temporal probe encoder
and frozen VAE decoder.
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
from matplotlib.animation import FuncAnimation, PillowWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

current_dir = Path(__file__).parent

# Load VAE model
spec = importlib.util.spec_from_file_location("model_vae", current_dir / "33_model_multiscale_vae_elbow.py")
model_vae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_vae)
MultiScaleVAE = model_vae.MultiScaleVAE

# Load probe encoder
spec = importlib.util.spec_from_file_location("probe_encoder", current_dir / "36_model_temporal_probe_encoder.py")
probe_encoder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_encoder_module)
TemporalProbeEncoder = probe_encoder_module.TemporalProbeEncoder
ProbeVAE = probe_encoder_module.ProbeVAE

# Load dataset
spec = importlib.util.spec_from_file_location("train_probe", current_dir / "37_train_probe_reconstruction.py")
train_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_probe)
ProbeReconstructionDataset = train_probe.ProbeReconstructionDataset


@torch.no_grad()
def visualize_reconstructions(model, dataset, device, output_dir, num_samples=5):
    """Create static comparison images."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample indices from the dataset
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    for idx, sample_idx in enumerate(indices):
        probe_seq, target_img = dataset[sample_idx]
        
        # Add batch dimension
        probe_seq = probe_seq.unsqueeze(0).to(device)
        target_img = target_img.unsqueeze(0).to(device)
        
        # Get reconstruction
        outputs, mu, logvar = model(probe_seq)
        
        # Create comparison plot
        # outputs is a dict {64: tensor, 128: tensor, 256: tensor}
        scales = sorted(outputs.keys())
        num_vars = target_img.shape[1]
        
        fig, axes = plt.subplots(num_vars, len(scales) + 1, figsize=(5 * (len(scales) + 1), 5 * num_vars))
        
        if num_vars == 1:
            axes = axes.reshape(1, -1)
        
        var_names = ['n', 'phi'] if num_vars == 2 else ['var']
        
        for v in range(num_vars):
            # Ground truth
            ax = axes[v, 0]
            im = ax.imshow(target_img[0, v].cpu().numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'Ground Truth ({var_names[v]})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Reconstructions at different scales
            for s_idx, scale in enumerate(scales):
                ax = axes[v, s_idx + 1]
                recon = outputs[scale][0, v].cpu().numpy()
                im = ax.imshow(recon, cmap='viridis', aspect='auto')
                ax.set_title(f'Reconstruction {scale}x{scale} ({var_names[v]})')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'reconstruction_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved reconstruction {idx+1}/{num_samples}")
    
    print(f"\nAll static visualizations saved to {output_dir}")


@torch.no_grad()
def create_reconstruction_animation(model, dataset, device, output_dir, num_frames=50):
    """Create animated comparison of ground truth vs reconstruction over time."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample frames evenly across the dataset
    frame_indices = np.linspace(0, len(dataset)-1, num_frames, dtype=int)
    
    num_vars = 2  # n, phi
    var_names = ['n', 'phi']
    target_scale = 512  # Use highest resolution reconstruction
    
    # Prepare data
    print("Preparing animation frames...")
    ground_truths = []
    reconstructions = []
    
    for frame_idx in frame_indices:
        probe_seq, target_img = dataset[frame_idx]
        probe_seq = probe_seq.unsqueeze(0).to(device)
        target_img = target_img.unsqueeze(0).to(device)
        
        outputs, _, _ = model(probe_seq)
        
        ground_truths.append(target_img[0].cpu().numpy())
        reconstructions.append(outputs[target_scale][0].cpu().numpy())
    
    # Create animation for each variable
    for v in range(num_vars):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize plots
        gt_data = ground_truths[0][v]
        recon_data = reconstructions[0][v]
        
        vmin = min(gt_data.min(), recon_data.min())
        vmax = max(gt_data.max(), recon_data.max())
        
        im1 = axes[0].imshow(gt_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Ground Truth ({var_names[v]})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(recon_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Probe Reconstruction ({var_names[v]})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        fig.suptitle(f'Frame 0/{num_frames}', fontsize=14)
        
        def update(frame):
            gt_data = ground_truths[frame][v]
            recon_data = reconstructions[frame][v]
            
            im1.set_data(gt_data)
            im2.set_data(recon_data)
            
            fig.suptitle(f'Frame {frame}/{num_frames}', fontsize=14)
            return im1, im2
        
        anim = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
        
        output_path = output_dir / f'animation_{var_names[v]}.gif'
        writer = PillowWriter(fps=10)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f"Saved animation for {var_names[v]} to {output_path}")
    
    print(f"\nAll animations saved to {output_dir}")


@torch.no_grad()
def analyze_reconstruction_quality(model, dataset, device, output_dir):
    """Compute and visualize comprehensive reconstruction quality metrics."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Computing reconstruction quality metrics...")
    
    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    metrics_per_scale = {64: [], 128: [], 256: [], 512: []}
    
    # Compute metrics over all samples
    num_samples = min(len(dataset), 100)  # Limit for speed
    for i in range(num_samples):
        probe_seq, target_img = dataset[i]
        probe_seq = probe_seq.unsqueeze(0).to(device)
        target_img = target_img.unsqueeze(0).to(device)
        
        outputs, _, _ = model(probe_seq)
        
        for scale, recon in outputs.items():
            # Downsample target to match scale
            if target_img.shape[-1] != scale:
                target_scaled = torch.nn.functional.interpolate(
                    target_img, size=(scale, scale), mode='bilinear', align_corners=False
                )
            else:
                target_scaled = target_img
            
            # Normalize to [0, 1] for metrics
            output_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
            target_norm = (target_scaled - target_scaled.min()) / (target_scaled.max() - target_scaled.min() + 1e-8)
            
            # MSE
            mse = torch.nn.functional.mse_loss(recon, target_scaled).item()
            
            # SSIM
            ssim_val = ssim_metric(output_norm, target_norm).item()
            
            # MS-SSIM (requires minimum size)
            if scale >= 160:
                ms_ssim_val = ms_ssim_metric(output_norm, target_norm).item()
            else:
                ms_ssim_val = None
            
            # LPIPS (requires 3 channels and [0, 1] range)
            if output_norm.shape[1] == 2:
                output_3ch = torch.cat([output_norm, output_norm[:, :1, :, :]], dim=1)
                target_3ch = torch.cat([target_norm, target_norm[:, :1, :, :]], dim=1)
            else:
                output_3ch = output_norm.repeat(1, 3, 1, 1)
                target_3ch = target_norm.repeat(1, 3, 1, 1)
            
            lpips_val = lpips_metric(output_3ch, target_3ch).item()
            
            metrics_per_scale[scale].append({
                'mse': mse,
                'ssim': ssim_val,
                'ms_ssim': ms_ssim_val,
                'lpips': lpips_val,
            })
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_samples} samples")
    
    # Aggregate metrics
    aggregated_metrics = {}
    for scale in sorted(metrics_per_scale.keys()):
        scale_metrics = metrics_per_scale[scale]
        aggregated_metrics[scale] = {
            'mse': np.mean([m['mse'] for m in scale_metrics]),
            'ssim': np.mean([m['ssim'] for m in scale_metrics]),
            'ms_ssim': np.mean([m['ms_ssim'] for m in scale_metrics if m['ms_ssim'] is not None]) if scale >= 160 else None,
            'lpips': np.mean([m['lpips'] for m in scale_metrics]),
        }
    
    # Save metrics to file
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Probe Reconstruction Metrics\n")
        f.write("=" * 60 + "\n\n")
        
        for scale in sorted(aggregated_metrics.keys()):
            f.write(f"Scale: {scale}x{scale}\n")
            f.write(f"  MSE:     {aggregated_metrics[scale]['mse']:.6f}\n")
            f.write(f"  SSIM:    {aggregated_metrics[scale]['ssim']:.6f}\n")
            if aggregated_metrics[scale]['ms_ssim'] is not None:
                f.write(f"  MS-SSIM: {aggregated_metrics[scale]['ms_ssim']:.6f}\n")
            f.write(f"  LPIPS:   {aggregated_metrics[scale]['lpips']:.6f}\n")
            f.write("\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scales = sorted(aggregated_metrics.keys())
    
    # MSE
    ax = axes[0, 0]
    mse_means = [aggregated_metrics[s]['mse'] for s in scales]
    ax.bar(range(len(scales)), mse_means)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s}x{s}' for s in scales])
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error by Scale')
    ax.grid(True, alpha=0.3)
    
    # SSIM
    ax = axes[0, 1]
    ssim_means = [aggregated_metrics[s]['ssim'] for s in scales]
    ax.bar(range(len(scales)), ssim_means)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s}x{s}' for s in scales])
    ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity Index by Scale')
    ax.grid(True, alpha=0.3)
    
    # MS-SSIM
    ax = axes[1, 0]
    ms_ssim_scales = [s for s in scales if aggregated_metrics[s]['ms_ssim'] is not None]
    ms_ssim_means = [aggregated_metrics[s]['ms_ssim'] for s in ms_ssim_scales]
    if ms_ssim_means:
        ax.bar(range(len(ms_ssim_scales)), ms_ssim_means)
        ax.set_xticks(range(len(ms_ssim_scales)))
        ax.set_xticklabels([f'{s}x{s}' for s in ms_ssim_scales])
    ax.set_ylabel('MS-SSIM')
    ax.set_title('Multi-Scale SSIM by Scale')
    ax.grid(True, alpha=0.3)
    
    # LPIPS
    ax = axes[1, 1]
    lpips_means = [aggregated_metrics[s]['lpips'] for s in scales]
    ax.bar(range(len(scales)), lpips_means)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s}x{s}' for s in scales])
    ax.set_ylabel('LPIPS')
    ax.set_title('Perceptual Similarity (LPIPS) by Scale')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reconstruction quality analysis saved to {output_dir / 'reconstruction_quality.png'}")
    
    # Print summary
    print("\nReconstruction Quality Summary:")
    print("=" * 60)
    for scale in scales:
        print(f"{scale}x{scale}:")
        print(f"  MSE:     {aggregated_metrics[scale]['mse']:.6f}")
        print(f"  SSIM:    {aggregated_metrics[scale]['ssim']:.6f}")
        if aggregated_metrics[scale]['ms_ssim'] is not None:
            print(f"  MS-SSIM: {aggregated_metrics[scale]['ms_ssim']:.6f}")
        print(f"  LPIPS:   {aggregated_metrics[scale]['lpips']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained probe reconstruction checkpoint")
    parser.add_argument("--sim-data-dir", type=str,
                       default="/dtu/blackhole/1b/223803/tcv_probe_data/simulation")
    parser.add_argument("--probe-data-dir", type=str,
                       default="/dtu/blackhole/1b/223803/tcv_probe_data/probes")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "phi"])
    parser.add_argument("--output", type=str,
                       default="/dtu/blackhole/1b/223803/results/probe_reconstruction")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of static samples to visualize")
    parser.add_argument("--num-frames", type=int, default=50,
                       help="Number of frames for animation")
    parser.add_argument("--create-animation", action="store_true",
                       help="Create animated GIFs")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PROBE RECONSTRUCTION INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint['args']
    
    # Load VAE
    print("Loading VAE...")
    vae_model = MultiScaleVAE(
        in_channels=len(args.variables),
        latent_dim=256,
        base_channels=32,
        target_sizes=[64, 128, 256, 512],
    ).to(device)
    
    # Load probe encoder
    print("Loading probe encoder...")
    probe_encoder = TemporalProbeEncoder(
        num_probes=saved_args['num_probes'],
        num_variables=len(args.variables),
        seq_len=saved_args['seq_len'],
        latent_dim=256,
        hidden_dim=512,
    )
    
    # Create combined model
    model = ProbeVAE(vae_model, probe_encoder).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ProbeReconstructionDataset(
        sim_data_path=Path(args.sim_data_dir),
        probe_data_path=Path(args.probe_data_dir),
        variables=args.variables,
        seq_len=saved_args['seq_len'],
        num_probes=saved_args['num_probes'],
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    visualize_reconstructions(model, dataset, device, output_dir, args.num_samples)
    
    if args.create_animation:
        create_reconstruction_animation(model, dataset, device, output_dir, args.num_frames)
    
    analyze_reconstruction_quality(model, dataset, device, output_dir)
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
