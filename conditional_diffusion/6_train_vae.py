#!/usr/bin/env python3
"""
Training script for VAE denoising model.

Trains a VAE to denoise tokamak plasma field data using ELBO loss.
"""

import argparse
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Import with actual filenames using importlib
import importlib.util

current_dir = Path(__file__).parent

# Load 4_dataset.py
spec = importlib.util.spec_from_file_location("dataset_4", current_dir / "4_dataset.py")
dataset_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_4)
build_dataloaders = dataset_4.build_dataloaders

# Load 5_model.py
spec = importlib.util.spec_from_file_location("model_5", current_dir / "5_model.py")
model_5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_5)
VAE = model_5.VAE
vae_loss = model_5.vae_loss
count_parameters = model_5.count_parameters


def train_epoch(
    model: VAE,
    dataloader,
    optimizer,
    device,
    kl_weight: float = 1e-4,
    use_amp: bool = False,
    scaler=None,
):
    """
    Train for one epoch.
    
    Returns
    -------
    avg_loss : float
        Average total loss
    avg_recon : float
        Average reconstruction loss
    avg_kl : float
        Average KL divergence
    """
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for clean, noisy in dataloader:
        clean = clean.to(device)
        noisy = noisy.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                x_recon, mu, logvar = model(noisy)
                loss, recon_loss, kl_loss = vae_loss(
                    x_recon, clean, mu, logvar, kl_weight
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_recon, mu, logvar = model(noisy)
            loss, recon_loss, kl_loss = vae_loss(
                x_recon, clean, mu, logvar, kl_weight
            )
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    
    return avg_loss, avg_recon, avg_kl


@torch.no_grad()
def validate(
    model: VAE,
    dataloader,
    device,
    kl_weight: float = 1e-4,
):
    """
    Validate model.
    
    Returns
    -------
    avg_loss : float
        Average total loss
    avg_recon : float
        Average reconstruction loss
    avg_kl : float
        Average KL divergence
    """
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for clean, noisy in dataloader:
        clean = clean.to(device)
        noisy = noisy.to(device)
        
        x_recon, mu, logvar = model(noisy)
        loss, recon_loss, kl_loss = vae_loss(
            x_recon, clean, mu, logvar, kl_weight
        )
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    
    return avg_loss, avg_recon, avg_kl


def fit(
    model: VAE,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    kl_weight: float = 1e-4,
    output_dir: Path = None,
    use_amp: bool = False,
    save_every: int = 10,
):
    """
    Training loop.
    
    Parameters
    ----------
    model : VAE
        Model to train
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    device : torch.device
        Device to train on
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    kl_weight : float
        Weight for KL divergence term
    output_dir : Path
        Directory to save checkpoints
    use_amp : bool
        Whether to use automatic mixed precision
    save_every : int
        Save checkpoint every N epochs
    """
    if output_dir is None:
        output_dir = Path("runs/vae_denoising")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    scaler = GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"KL weight: {kl_weight}")
    print(f"Mixed precision: {use_amp}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, kl_weight, use_amp, scaler
        )
        
        # Validate
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, kl_weight
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}) | "
            f"Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}) | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE for denoising")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/bout_data",
        help="Directory with extracted .npy files"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "te", "ti", "phi"],
        help="Variables to train on"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Noise standard deviation (or min if --noise-max set)"
    )
    parser.add_argument(
        "--noise-max",
        type=float,
        default=None,
        help="Max noise level for random multi-level training"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize images to this size (e.g., 256 for 256x256)"
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Load data on-demand to save memory"
    )
    parser.add_argument(
        "--probe-augmentation-prob",
        type=float,
        default=0.4,
        help="Probability of using probe-based interpolation augmentation"
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=40,
        help="Number of random probe points for interpolation augmentation"
    )
    
    # Model arguments
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension"
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base number of channels"
    )
    parser.add_argument(
        "--intermediate-noise-scale",
        type=float,
        default=0.0,
        help="Noise scale for intermediate decoder stages (0 = disabled, try 0.01-0.1)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-4,
        help="Weight for KL divergence term"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/runs/vae_denoising",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Handle noise level range
    if args.noise_max is not None:
        noise_level = (args.noise_level, args.noise_max)
        print(f"Using multi-level noise: [{args.noise_level}, {args.noise_max}]")
    else:
        noise_level = args.noise_level
    
    # Determine input size for model
    input_size = args.resize if args.resize else 1024
    
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        noise_level=noise_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize=args.resize,
        lazy_load=args.lazy_load,
        probe_augmentation_prob=args.probe_augmentation_prob,
        num_probes=args.num_probes,
    )
    
    print("\nBuilding model...")
    model = VAE(
        in_channels=len(args.variables),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        input_size=input_size,
        intermediate_noise_scale=args.intermediate_noise_scale,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    if args.intermediate_noise_scale > 0:
        print(f"Intermediate noise injection: {args.intermediate_noise_scale} (during training only)")
    
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        output_dir=args.output,
        use_amp=args.use_amp,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
