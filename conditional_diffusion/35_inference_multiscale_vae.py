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

def visualize_reconstructions(model, dataloader, device, output_dir, num_samples=5):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch of data
    clean, _ = next(iter(dataloader))
    clean = clean.to(device)
    
    with torch.no_grad():
        outputs, mu, logvar = model(clean)
    
    # outputs is a dict {64: tensor, 128: tensor, 256: tensor}
    
    # Plotting
    for i in range(min(num_samples, clean.size(0))):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original Input (256x256)
        # Channel 0 (n)
        im0 = axes[0, 0].imshow(clean[i, 0].cpu().numpy(), cmap='viridis')
        axes[0, 0].set_title("Original (n)")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Channel 1 (phi)
        im1 = axes[1, 0].imshow(clean[i, 1].cpu().numpy(), cmap='viridis')
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
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "phi"])
    parser.add_argument("--output", type=str, default="results/multiscale_vae_elbow")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = MultiScaleVAE(
        in_channels=len(args.variables),
        latent_dim=256,
        base_channels=32,
        target_sizes=[64, 128, 256],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (Epoch {checkpoint['epoch']})")
    
    # Load Data
    _, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        noise_level=0.0,
        batch_size=8,
        num_workers=4,
        resize=256,
        lazy_load=True,
    )
    
    visualize_reconstructions(model, val_loader, device, args.output)

if __name__ == "__main__":
    main()
