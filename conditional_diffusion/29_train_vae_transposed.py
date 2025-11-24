#!/usr/bin/env python3
"""
Training script for VAE with transposed convolutions and heavy augmentation.
"""

import argparse
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import importlib.util

current_dir = Path(__file__).parent

# Load dataset
spec = importlib.util.spec_from_file_location("dataset_28", current_dir / "28_dataset_heavy_augmentation.py")
dataset_28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_28)
build_dataloaders = dataset_28.build_dataloaders

# Load model
spec = importlib.util.spec_from_file_location("model_27", current_dir / "27_model_vae_transposed.py")
model_27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_27)
VAETransposed = model_27.VAETransposed
vae_loss = model_27.vae_loss
count_parameters = model_27.count_parameters


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    kl_weight: float = 1e-4,
    use_amp: bool = False,
    scaler=None,
):
    """Train for one epoch."""
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
    
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches


@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    kl_weight: float = 1e-4,
):
    """Validate model."""
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
    
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches


def fit(
    model,
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
    """Training loop."""
    if output_dir is None:
        output_dir = Path("runs/vae_transposed")
    
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
        
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, kl_weight, use_amp, scaler
        )
        
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, kl_weight
        )
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}) | "
            f"Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}) | "
            f"Time: {epoch_time:.1f}s"
        )
        
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
        
        if epoch % save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE with transposed convs and heavy augmentation")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--noise-max", type=float, default=None)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument("--samples-per-image", type=int, default=5, help="Augmentation multiplier (5x data)")
    parser.add_argument("--num-probes-min", type=int, default=20, help="Min number of probe points")
    parser.add_argument("--num-probes-max", type=int, default=60, help="Max number of probe points")
    parser.add_argument("--probe-noise-std", type=float, default=0.05, help="Noise on probe values")
    
    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--intermediate-noise-scale", type=float, default=0.0)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/runs/vae_transposed_heavy")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Handle noise level range
    if args.noise_max is not None:
        noise_level = (args.noise_level, args.noise_max)
        print(f"Using multi-level noise: [{args.noise_level}, {args.noise_max}]")
    else:
        noise_level = args.noise_level
    
    # Determine input size
    input_size = args.resize if args.resize else 1024
    
    print("Building dataloaders with heavy augmentation...")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        noise_level=noise_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize=args.resize,
        lazy_load=args.lazy_load,
        samples_per_image=args.samples_per_image,
        num_probes_range=(args.num_probes_min, args.num_probes_max),
        probe_noise_std=args.probe_noise_std,
    )
    
    print("\nBuilding VAE with transposed convolutions...")
    model = VAETransposed(
        in_channels=len(args.variables),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        input_size=input_size,
        intermediate_noise_scale=args.intermediate_noise_scale,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    if args.intermediate_noise_scale > 0:
        print(f"Intermediate noise injection: {args.intermediate_noise_scale}")
    
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
