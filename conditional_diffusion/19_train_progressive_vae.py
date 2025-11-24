#!/usr/bin/env python3
"""
Training script for Progressive Multi-Scale VAE.

Trains VAE with:
- Progressive upsampling
- Residual refinement
- Multi-scale loss
"""

import argparse
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Import with importlib
import importlib.util

current_dir = Path(__file__).parent

# Load dataset (reuse existing)
spec = importlib.util.spec_from_file_location("dataset_4", current_dir / "4_dataset.py")
dataset_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_4)
build_dataloaders = dataset_4.build_dataloaders

# Load progressive VAE model
spec = importlib.util.spec_from_file_location("model_progressive", current_dir / "18_model_progressive_vae.py")
model_progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_progressive)
ProgressiveVAE = model_progressive.ProgressiveVAE
progressive_vae_loss = model_progressive.progressive_vae_loss
count_parameters = model_progressive.count_parameters


def train_epoch(
    model: ProgressiveVAE,
    dataloader,
    optimizer,
    device,
    kl_weight: float = 1e-4,
    scale_weights: dict = None,
    use_amp: bool = False,
    scaler=None,
):
    """Train for one epoch with multi-scale loss."""
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
                outputs, mu, logvar = model(noisy)
                loss, recon_loss, kl_loss, _ = progressive_vae_loss(
                    outputs, clean, mu, logvar, kl_weight, scale_weights
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, mu, logvar = model(noisy)
            loss, recon_loss, kl_loss, _ = progressive_vae_loss(
                outputs, clean, mu, logvar, kl_weight, scale_weights
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
    model: ProgressiveVAE,
    dataloader,
    device,
    kl_weight: float = 1e-4,
    scale_weights: dict = None,
):
    """Validate model with multi-scale loss."""
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    # Track per-scale losses
    all_scale_losses = {size: 0.0 for size in model.target_sizes}
    
    for clean, noisy in dataloader:
        clean = clean.to(device)
        noisy = noisy.to(device)
        
        outputs, mu, logvar = model(noisy)
        loss, recon_loss, kl_loss, scale_losses = progressive_vae_loss(
            outputs, clean, mu, logvar, kl_weight, scale_weights
        )
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        for size, scale_loss in scale_losses.items():
            all_scale_losses[size] += scale_loss.item()
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    
    avg_scale_losses = {size: loss / num_batches for size, loss in all_scale_losses.items()}
    
    return avg_loss, avg_recon, avg_kl, avg_scale_losses


def fit(
    model: ProgressiveVAE,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    kl_weight: float = 1e-4,
    scale_weights: dict = None,
    output_dir: Path = None,
    use_amp: bool = False,
    save_every: int = 10,
):
    """Training loop."""
    if output_dir is None:
        output_dir = Path("runs/progressive_vae")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    scaler = GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    
    print(f"\nStarting Progressive VAE training for {epochs} epochs")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Progressive scales: {model.target_sizes}")
    print(f"Scale weights: {scale_weights if scale_weights else 'Equal'}")
    print(f"Learning rate: {lr}")
    print(f"KL weight: {kl_weight}")
    print(f"Intermediate noise: {model.intermediate_noise_scale}")
    print(f"Mixed precision: {use_amp}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, 
            kl_weight, scale_weights, use_amp, scaler
        )
        
        # Validate
        val_loss, val_recon, val_kl, val_scale_losses = validate(
            model, val_loader, device, kl_weight, scale_weights
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        scale_loss_str = ", ".join([f"{size}={loss:.4f}" for size, loss in val_scale_losses.items()])
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}) | "
            f"Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}) | "
            f"Scales: [{scale_loss_str}] | "
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
                'target_sizes': model.target_sizes,
            }, best_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        
        # Periodic checkpoint
        if epoch % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'target_sizes': model.target_sizes,
            }, ckpt_path)
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Progressive Multi-Scale VAE")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--noise-max", type=float, default=None)
    parser.add_argument("--resize", type=int, default=256, 
                       help="Input size (will also determine max target size)")
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument("--probe-augmentation-prob", type=float, default=0.4)
    parser.add_argument("--num-probes", type=int, default=40)
    
    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--target-sizes", type=int, nargs="+", default=[64, 128, 256],
                       help="Progressive output sizes (e.g., 64 128 256)")
    parser.add_argument("--intermediate-noise-scale", type=float, default=0.0)
    
    # Loss weights
    parser.add_argument("--scale-weights", type=str, default=None,
                       help="Scale weights as 'size1:weight1,size2:weight2' (e.g., '64:0.5,128:1.0,256:2.0')")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/runs/progressive_vae")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Parse scale weights if provided
    scale_weights = None
    if args.scale_weights:
        scale_weights = {}
        for pair in args.scale_weights.split(','):
            size_str, weight_str = pair.split(':')
            scale_weights[int(size_str)] = float(weight_str)
        print(f"Using custom scale weights: {scale_weights}")
    
    # Handle noise level
    if args.noise_max is not None:
        noise_level = (args.noise_level, args.noise_max)
    else:
        noise_level = args.noise_level
    
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
    
    print("\nBuilding Progressive VAE...")
    model = ProgressiveVAE(
        in_channels=len(args.variables),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        target_sizes=args.target_sizes,
        intermediate_noise_scale=args.intermediate_noise_scale,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        scale_weights=scale_weights,
        output_dir=args.output,
        use_amp=args.use_amp,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
