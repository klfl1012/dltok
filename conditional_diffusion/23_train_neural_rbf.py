#!/usr/bin/env python3
"""
Training script for Neural RBF.
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import importlib.util

current_dir = Path(__file__).parent

# Load Neural RBF model
spec = importlib.util.spec_from_file_location("model_rbf", current_dir / "21_model_neural_rbf.py")
model_rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_rbf)
NeuralRBF = model_rbf.NeuralRBF
neural_rbf_loss = model_rbf.neural_rbf_loss
count_parameters = model_rbf.count_parameters


class ProbeDataset(Dataset):
    """Dataset for probe-to-field reconstruction."""
    
    def __init__(
        self,
        probe_dir: Path,
        data_dir: Path,
        probe_coords: np.ndarray,
        variables: list,
        spatial_size: int = 256,
        train: bool = True,
        train_fraction: float = 0.8,
        normalize: bool = True,
    ):
        self.probe_dir = Path(probe_dir)
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.spatial_size = spatial_size
        self.normalize = normalize
        
        # Load data
        self.probe_data = {}
        self.field_data = {}
        self.stats = {}
        
        # Normalize probe coordinates to [0, 1]
        self.probe_coords_norm = probe_coords.copy().astype(np.float32)
        self.probe_coords_norm[:, 0] /= 1024  # x
        self.probe_coords_norm[:, 1] /= 1024  # z
        
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
        
        if train:
            self.indices = np.arange(0, split_idx)
        else:
            self.indices = np.arange(split_idx, total_timesteps)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]
        
        # Stack probe values
        probe_vals = np.stack([self.probe_data[var][t] for var in self.variables], axis=0)
        
        # Stack fields and resize
        fields = []
        for var in self.variables:
            field = self.field_data[var][t]
            field_t = torch.from_numpy(field).unsqueeze(0).float()
            field_resized = torch.nn.functional.interpolate(
                field_t.unsqueeze(0), 
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            fields.append(field_resized)
        
        field_vals = torch.cat(fields, dim=0)
        probe_vals = torch.from_numpy(probe_vals).float()
        
        return probe_vals, field_vals


def train_epoch(model, dataloader, probe_coords, optimizer, device, rbf_weight, use_amp, scaler):
    model.train()
    total_loss = 0.0
    total_rbf = 0.0
    total_ref = 0.0
    num_batches = 0
    
    for probe_vals, field_vals in dataloader:
        probe_vals = probe_vals.to(device)
        field_vals = field_vals.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                rbf_out, ref_out = model(probe_vals, probe_coords)
                loss, rbf_l, ref_l = neural_rbf_loss(rbf_out, ref_out, field_vals, rbf_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            rbf_out, ref_out = model(probe_vals, probe_coords)
            loss, rbf_l, ref_l = neural_rbf_loss(rbf_out, ref_out, field_vals, rbf_weight)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_rbf += rbf_l.item()
        total_ref += ref_l.item()
        num_batches += 1
    
    return total_loss / num_batches, total_rbf / num_batches, total_ref / num_batches


@torch.no_grad()
def validate(model, dataloader, probe_coords, device, rbf_weight):
    model.eval()
    total_loss = 0.0
    total_rbf = 0.0
    total_ref = 0.0
    num_batches = 0
    
    for probe_vals, field_vals in dataloader:
        probe_vals = probe_vals.to(device)
        field_vals = field_vals.to(device)
        
        rbf_out, ref_out = model(probe_vals, probe_coords)
        loss, rbf_l, ref_l = neural_rbf_loss(rbf_out, ref_out, field_vals, rbf_weight)
        
        total_loss += loss.item()
        total_rbf += rbf_l.item()
        total_ref += ref_l.item()
        num_batches += 1
    
    return total_loss / num_batches, total_rbf / num_batches, total_ref / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=str, default="/dtu/blackhole/1b/223803/probe_data")
    parser.add_argument("--data-dir", type=str, default="/dtu/blackhole/1b/223803/bout_data")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "te", "ti", "phi"])
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--num-rbf-centers", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--rbf-weight", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/dtu/blackhole/1b/223803/runs/neural_rbf")
    parser.add_argument("--use-amp", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load probe coordinates
    probe_coords = np.load(Path(args.probe_dir) / "probe_coordinates.npy")
    probe_coords_norm = probe_coords.copy().astype(np.float32)
    probe_coords_norm[:, 0] /= 1024
    probe_coords_norm[:, 1] /= 1024
    probe_coords_t = torch.from_numpy(probe_coords_norm).to(device)
    
    print(f"Loaded {len(probe_coords)} probe coordinates")
    
    # Create datasets
    train_dataset = ProbeDataset(
        args.probe_dir, args.data_dir, probe_coords, args.variables, 
        args.spatial_size, train=True
    )
    val_dataset = ProbeDataset(
        args.probe_dir, args.data_dir, probe_coords, args.variables,
        args.spatial_size, train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = NeuralRBF(
        num_probes=len(probe_coords),
        num_vars=len(args.variables),
        num_rbf_centers=args.num_rbf_centers,
        spatial_size=(args.spatial_size, args.spatial_size),
        base_channels=args.base_channels,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler() if args.use_amp else None
    
    best_val_loss = float('inf')
    
    print(f"\nStarting Neural RBF training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_rbf, train_ref = train_epoch(
            model, train_loader, probe_coords_t, optimizer, device, 
            args.rbf_weight, args.use_amp, scaler
        )
        val_loss, val_rbf, val_ref = validate(
            model, val_loader, probe_coords_t, device, args.rbf_weight
        )
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} (RBF={train_rbf:.4f}, Ref={train_ref:.4f}) | "
              f"Val: {val_loss:.4f} (RBF={val_rbf:.4f}, Ref={val_ref:.4f}) | "
              f"{elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model")
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
