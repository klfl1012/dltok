#!/usr/bin/env python3
"""
Training script for Multi-Scale VAE with Elbow Loss.
"""

import argparse
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

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
elbow_loss = model_vae.elbow_loss
mse_loss = model_vae.mse_loss
count_parameters = model_vae.count_parameters


def train_epoch(model, dataloader, optimizer, device, kl_weight, scale_weights, use_amp, scaler, loss_type='elbow'):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for data in dataloader:
        data = data.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, mu, logvar = model(data)
                if loss_type == 'mse':
                    loss, recon_loss, kl_loss, _ = mse_loss(
                        outputs, data, mu, logvar, kl_weight, scale_weights
                    )
                else:
                    loss, recon_loss, kl_loss, _ = elbow_loss(
                        outputs, data, mu, logvar, kl_weight, scale_weights
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, mu, logvar = model(data)
            if loss_type == 'mse':
                loss, recon_loss, kl_loss, _ = mse_loss(
                    outputs, data, mu, logvar, kl_weight, scale_weights
                )
            else:
                loss, recon_loss, kl_loss, _ = elbow_loss(
                    outputs, data, mu, logvar, kl_weight, scale_weights
                )
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches


@torch.no_grad()
def validate(model, dataloader, device, kl_weight, scale_weights, loss_type='elbow'):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    all_scale_losses = {}
    
    for data in dataloader:
        data = data.to(device)
        
        outputs, mu, logvar = model(data)
        if loss_type == 'mse':
            loss, recon_loss, kl_loss, scale_losses = mse_loss(
                outputs, data, mu, logvar, kl_weight, scale_weights
            )
        else:
            loss, recon_loss, kl_loss, scale_losses = elbow_loss(
                outputs, data, mu, logvar, kl_weight, scale_weights
            )
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        for size, scale_loss in scale_losses.items():
            if size not in all_scale_losses:
                all_scale_losses[size] = 0.0
            all_scale_losses[size] += scale_loss.item()
        
        num_batches += 1
    
    avg_scale_losses = {size: loss / num_batches for size, loss in all_scale_losses.items()}
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches, avg_scale_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/tcv_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "phi"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4, help="KL divergence weight")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/runs/multiscale_vae_elbow")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Model architecture options
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent space dimension")
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels")
    parser.add_argument("--loss-type", type=str, default="elbow", choices=["elbow", "mse"], help="Reconstruction loss type")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Multi-Scale VAE with Elbow Loss")
    print(f"Data directory: {args.data_dir}")
    print(f"Variables: {args.variables}")
    
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    model = MultiScaleVAE(
        in_channels=len(args.variables),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        target_sizes=[64, 128, 256, 512],
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Configuration: latent_dim={args.latent_dim}, base_channels={args.base_channels}, loss={args.loss_type}, kl_weight={args.kl_weight}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler(device='cuda') if args.use_amp else None
    
    scale_weights = {64: 0.5, 128: 1.0, 256: 2.0, 512: 4.0}
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Set loss type
    loss_type = args.loss_type
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, args.kl_weight, scale_weights, args.use_amp, scaler, loss_type
        )
        
        val_loss, val_recon, val_kl, val_scale_losses = validate(
            model, val_loader, device, args.kl_weight, scale_weights, loss_type
        )
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        scale_str = ", ".join([f"{k}={v:.4f}" for k, v in val_scale_losses.items()])
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}) | Scales: [{scale_str}] | {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / "best_model.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={args.patience})")
                break

if __name__ == "__main__":
    main()
