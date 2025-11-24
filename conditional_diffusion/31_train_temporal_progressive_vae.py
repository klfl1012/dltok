#!/usr/bin/env python3
"""
Training script for Temporal Progressive VAE.

Trains VAE with:
- ConvLSTM temporal encoder (processes 10 timesteps of probes)
- Progressive multi-scale decoder
- Reconstructs current frame from probe history
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.interpolate import Rbf, interp1d

# Import with importlib
import importlib.util

current_dir = Path(__file__).parent

# Load progressive VAE model
spec = importlib.util.spec_from_file_location("model_progressive", current_dir / "18_model_progressive_vae.py")
model_progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_progressive)
TemporalProgressiveVAE = model_progressive.TemporalProgressiveVAE
progressive_vae_loss = model_progressive.progressive_vae_loss
count_parameters = model_progressive.count_parameters


class TemporalProbeDataset(Dataset):
    """
    Dataset for temporal probe-to-field reconstruction.
    
    Returns:
    - Probe sequence: (T, C, H, W) - past 10 timesteps of probe data
    - Target field: (C, H, W) - current timestep ground truth
    """
    
    def __init__(
        self,
        probe_dir: Path,
        data_dir: Path,
        probe_coords: np.ndarray,
        variables: list,
        sequence_length: int = 10,
        spatial_size: int = 256,
        train: bool = True,
        train_fraction: float = 0.8,
        normalize: bool = True,
        interpolate_probes: bool = True,
    ):
        self.probe_dir = Path(probe_dir)
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.sequence_length = sequence_length
        self.spatial_size = spatial_size
        self.normalize = normalize
        self.interpolate_probes = interpolate_probes
        
        # Normalize probe coordinates to [0, 1]
        self.probe_coords_norm = probe_coords.copy().astype(np.float32)
        self.probe_coords_norm[:, 0] /= 1024  # x
        self.probe_coords_norm[:, 1] /= 1024  # z
        
        # Scale to target size
        self.probe_coords_scaled = self.probe_coords_norm.copy()
        self.probe_coords_scaled[:, 0] *= spatial_size
        self.probe_coords_scaled[:, 1] *= spatial_size
        
        # Load data
        self.probe_data = {}
        self.field_data = {}
        self.stats = {}
        
        for var in variables:
            probe_path = self.probe_dir / f"probe_{var}.npy"
            field_path = self.data_dir / f"{var}.npy"
            
            probe_vals = np.load(probe_path)
            field_vals = np.load(field_path)
            
            mean = field_vals.mean()
            std = field_vals.std()
            self.stats[var] = {'mean': mean, 'std': std}
            
            if normalize:
                probe_vals = (probe_vals - mean) / (std + 1e-8)
                field_vals = (field_vals - mean) / (std + 1e-8)
            
            self.probe_data[var] = probe_vals
            self.field_data[var] = field_vals
        
        # Train/val split
        total_timesteps = probe_vals.shape[0]
        split_idx = int(total_timesteps * train_fraction)
        
        # Valid indices: need sequence_length history
        if train:
            self.indices = np.arange(sequence_length, split_idx)
        else:
            self.indices = np.arange(max(split_idx, sequence_length), total_timesteps)
    
    def create_probe_image_interpolated(self, probe_values: np.ndarray) -> np.ndarray:
        """Create probe image with RBF interpolation (robust to colinear points)."""
        from scipy.interpolate import Rbf
        
        H, W = self.spatial_size, self.spatial_size
        
        # Check if points are colinear (all on same line)
        unique_x = np.unique(self.probe_coords_scaled[:, 0])
        unique_z = np.unique(self.probe_coords_scaled[:, 1])
        
        if len(unique_x) == 1 or len(unique_z) == 1:
            # Fallback to 1D interpolation
            if len(unique_z) == 1:
                # Horizontal line - interpolate along x
                from scipy.interpolate import interp1d
                probe_z = int(unique_z[0])
                probe_x = self.probe_coords_scaled[:, 0]
                
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
                    
                    # Extend vertically with decay
                    for z in range(H):
                        distance = abs(z - probe_z)
                        decay = np.exp(-distance / (H * 0.1))
                        image[z, :] = interpolated_line * decay
            else:
                # Vertical line - interpolate along z
                from scipy.interpolate import interp1d
                probe_x = int(unique_x[0])
                probe_z = self.probe_coords_scaled[:, 1]
                
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
                    
                    # Extend horizontally with decay
                    for x in range(W):
                        distance = abs(x - probe_x)
                        decay = np.exp(-distance / (W * 0.1))
                        image[:, x] = interpolated_line * decay
        else:
            # 2D RBF interpolation (robust to almost-colinear points)
            try:
                rbf = Rbf(
                    self.probe_coords_scaled[:, 0],
                    self.probe_coords_scaled[:, 1],
                    probe_values,
                    function='linear',
                    smooth=0.1  # Small smoothing for stability
                )
                
                grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(H))
                image = rbf(grid_x, grid_z)
            except Exception as e:
                # Ultimate fallback: sparse image
                print(f"Warning: RBF interpolation failed ({e}), using sparse representation")
                image = self.create_probe_image_sparse(probe_values, noise_std=0.0)
        
        return image
    
    def create_probe_image_sparse(self, probe_values: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Create sparse probe image with noise background."""
        H, W = self.spatial_size, self.spatial_size
        image = np.random.randn(H, W) * noise_std
        
        for (x, z), value in zip(self.probe_coords_scaled.astype(int), probe_values):
            if 0 <= x < W and 0 <= z < H:
                image[int(z), int(x)] = value
        
        return image
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t_current = self.indices[idx]
        t_start = t_current - self.sequence_length
        
        # Build probe sequence
        probe_sequence = []
        for t in range(t_start, t_current):
            frame_probes = []
            for var in self.variables:
                probe_vals = self.probe_data[var][t]
                
                if self.interpolate_probes:
                    probe_img = self.create_probe_image_interpolated(probe_vals)
                else:
                    probe_img = self.create_probe_image_sparse(probe_vals)
                
                frame_probes.append(probe_img)
            
            probe_sequence.append(np.stack(frame_probes, axis=0))  # (C, H, W)
        
        probe_sequence = np.stack(probe_sequence, axis=0)  # (T, C, H, W)
        
        # Get target (current frame)
        target_fields = []
        for var in self.variables:
            field = self.field_data[var][t_current]
            
            # Resize to target size
            field_t = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
            field_resized = F.interpolate(
                field_t,
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            target_fields.append(field_resized)
        
        target = torch.stack(target_fields, dim=0)  # (C, H, W)
        probe_sequence = torch.from_numpy(probe_sequence).float()  # (T, C, H, W)
        
        return probe_sequence, target


def train_epoch(model, dataloader, optimizer, device, kl_weight, scale_weights, use_amp, scaler):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for probe_seq, target in dataloader:
        probe_seq = probe_seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, mu, logvar = model(probe_seq)
                loss, recon_loss, kl_loss, _ = progressive_vae_loss(
                    outputs, target, mu, logvar, kl_weight, scale_weights
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, mu, logvar = model(probe_seq)
            loss, recon_loss, kl_loss, _ = progressive_vae_loss(
                outputs, target, mu, logvar, kl_weight, scale_weights
            )
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches


@torch.no_grad()
def validate(model, dataloader, device, kl_weight, scale_weights):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    all_scale_losses = {}
    
    for probe_seq, target in dataloader:
        probe_seq = probe_seq.to(device)
        target = target.to(device)
        
        outputs, mu, logvar = model(probe_seq)
        loss, recon_loss, kl_loss, scale_losses = progressive_vae_loss(
            outputs, target, mu, logvar, kl_weight, scale_weights
        )
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        for size, scale_loss in scale_losses.items():
            if size not in all_scale_losses:
                all_scale_losses[size] = 0.0
            all_scale_losses[size] += scale_loss.item()
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    avg_scale_losses = {size: loss / num_batches for size, loss in all_scale_losses.items()}
    
    return avg_loss, avg_recon, avg_kl, avg_scale_losses


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Progressive VAE")
    
    # Data arguments
    parser.add_argument("--probe-dir", type=str, default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of past timesteps")
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--interpolate-probes", action="store_true", default=True)
    
    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--target-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--intermediate-noise-scale", type=float, default=0.0)
    
    # Loss weights
    parser.add_argument("--scale-weights", type=str, default=None)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/runs/temporal_progressive_vae")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse scale weights
    scale_weights = None
    if args.scale_weights:
        scale_weights = {}
        for pair in args.scale_weights.split(','):
            size_str, weight_str = pair.split(':')
            scale_weights[int(size_str)] = float(weight_str)
    
    # Load probe coordinates
    probe_coords = np.load(Path(args.probe_dir) / "probe_coordinates.npy")
    print(f"Loaded {len(probe_coords)} probe coordinates")
    
    # Create datasets
    train_dataset = TemporalProbeDataset(
        args.probe_dir, args.data_dir, probe_coords, args.variables,
        sequence_length=args.sequence_length,
        spatial_size=args.spatial_size,
        train=True,
        interpolate_probes=args.interpolate_probes,
    )
    
    val_dataset = TemporalProbeDataset(
        args.probe_dir, args.data_dir, probe_coords, args.variables,
        sequence_length=args.sequence_length,
        spatial_size=args.spatial_size,
        train=False,
        interpolate_probes=args.interpolate_probes,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = TemporalProgressiveVAE(
        in_channels=len(args.variables),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        target_sizes=args.target_sizes,
        hidden_dims=args.hidden_dims,
        sequence_length=args.sequence_length,
        intermediate_noise_scale=args.intermediate_noise_scale,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Sequence length: {model.sequence_length}")
    print(f"Progressive scales: {model.target_sizes}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler(device='cuda') if args.use_amp else None
    
    best_val_loss = float('inf')
    
    print(f"\nStarting Temporal Progressive VAE training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device,
            args.kl_weight, scale_weights, args.use_amp, scaler
        )
        
        val_loss, val_recon, val_kl, val_scale_losses = validate(
            model, val_loader, device, args.kl_weight, scale_weights
        )
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        scale_loss_str = ", ".join([f"{size}={loss:.4f}" for size, loss in val_scale_losses.items()])
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}) | "
            f"Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}) | "
            f"Scales: [{scale_loss_str}] | "
            f"{elapsed:.1f}s"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'target_sizes': model.target_sizes,
                'sequence_length': model.sequence_length,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model")
        
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'target_sizes': model.target_sizes,
                'sequence_length': model.sequence_length,
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
