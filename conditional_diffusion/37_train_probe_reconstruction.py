#!/usr/bin/env python3
"""
Training script for Probe-based Reconstruction using frozen VAE decoder.

Trains a temporal probe encoder to map probe measurements to the VAE latent space,
then uses the frozen VAE decoder to reconstruct the full field.
"""

import argparse
from pathlib import Path
import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import importlib.util
import json

current_dir = Path(__file__).parent

# Load VAE model
spec = importlib.util.spec_from_file_location("model_vae", current_dir / "33_model_multiscale_vae_elbow.py")
model_vae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_vae)
MultiScaleVAE = model_vae.MultiScaleVAE
elbow_loss = model_vae.elbow_loss

# Load probe encoder
spec = importlib.util.spec_from_file_location("probe_encoder", current_dir / "36_model_temporal_probe_encoder.py")
probe_encoder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_encoder_module)
TemporalProbeEncoder = probe_encoder_module.TemporalProbeEncoder
ProbeVAE = probe_encoder_module.ProbeVAE


class ProbeReconstructionDataset(Dataset):
    """
    Dataset that provides probe measurements and corresponding full field images.
    """
    def __init__(
        self,
        sim_data_path: Path,
        probe_data_path: Path,
        variables: list[str],
        seq_len: int = 10,
        num_probes: int = 64,
    ):
        """
        Parameters
        ----------
        sim_data_path : Path
            Path to simulation data directory (contains n.npy, phi.npy)
        probe_data_path : Path
            Path to probe data directory (contains probe_0/, probe_1/, ...)
        variables : list of str
            Variables to use (e.g., ['n', 'phi'])
        seq_len : int
            Number of past timesteps to use for reconstruction
        num_probes : int
            Number of probes
        """
        self.sim_data_path = Path(sim_data_path)
        self.probe_data_path = Path(probe_data_path)
        self.variables = variables
        self.seq_len = seq_len
        self.num_probes = num_probes
        
        # Load global statistics for normalization
        stats_path = sim_data_path / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded global statistics from {stats_path}")
        else:
            print(f"WARNING: No statistics file found at {stats_path}")
            print("Will compute per-sample normalization (not recommended)")
            self.stats = None
        
        print(f"Loading simulation data from {sim_data_path}")
        # Load simulation data - shape should be (T, X, Y, Z) or (T, H, W) after extraction
        self.sim_data = {}
        for var in variables:
            var_path = sim_data_path / f"{var}.npy"
            data = np.load(var_path, mmap_mode='r')
            print(f"  {var}: original shape = {data.shape}")
            
            # Reshape from (T, 514, 1, 512) to (T, 512, 512)
            # Remove dimension at index 1 (514 -> skip first 2 entries)
            # Remove dimension at index 2 (1 -> squeeze)
            if len(data.shape) == 4:
                # Assume shape is (T, X, 1, Y)
                # Take X from index 2 onwards (skip first 2)
                data = data[:, 2:, 0, :]  # (T, 512, 512)
                print(f"  {var}: reshaped to {data.shape}")
            elif len(data.shape) == 3:
                # If already (T, X, Y), just skip first 2 in X
                data = data[:, 2:, :]
                print(f"  {var}: reshaped to {data.shape}")
            
            self.sim_data[var] = data
        
        # Get time dimension
        self.num_timesteps = self.sim_data[variables[0]].shape[0]
        
        print(f"\nLoading probe data from {probe_data_path}")
        # Load probe data - organized as probe_X/varY.npy
        # Files are named like "n3.npy", "phi3.npy" for probe 3
        self.probe_data = {}
        loaded_count = 0
        for p in range(num_probes):
            self.probe_data[p] = {}
            probe_dir = probe_data_path / f"probe_{p}"
            
            if not probe_dir.exists():
                print(f"  WARNING: {probe_dir} does not exist")
                continue
            
            probe_loaded = True
            for var in variables:
                # Probe variable file is named like "n3.npy", "phi3.npy" for probe 3
                var_file = probe_dir / f"{var}{p}.npy"
                if var_file.exists():
                    data = np.load(var_file, mmap_mode='r')
                    # Store with simple variable name as key
                    self.probe_data[p][var] = data
                else:
                    print(f"  WARNING: {var_file} not found")
                    probe_loaded = False
            
            if probe_loaded:
                loaded_count += 1
        
        print(f"  Successfully loaded {loaded_count}/{num_probes} probes")
        
        # Check what keys we actually have for probe 0
        if 0 in self.probe_data and self.probe_data[0]:
            sample_keys = list(self.probe_data[0].keys())
            print(f"  Sample probe 0 keys: {sample_keys}")
            if sample_keys:
                print(f"  Sample probe data shape (probe 0, {sample_keys[0]}): {self.probe_data[0][sample_keys[0]].shape}")
        else:
            print("  WARNING: No data loaded for probe 0")
        
        if loaded_count == 0:
            raise RuntimeError("No probe data could be loaded! Check paths and file structure.")
        
        # Valid indices: need seq_len past frames
        self.valid_indices = list(range(seq_len, self.num_timesteps))
        print(f"\nDataset: {len(self.valid_indices)} valid samples (from t={seq_len} to t={self.num_timesteps-1})")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns probe sequence and target image.
        
        Returns
        -------
        probe_seq : torch.Tensor
            Shape: (seq_len, num_probes, num_variables)
        target_img : torch.Tensor
            Shape: (num_variables, H, W)
        """
        t = self.valid_indices[idx]
        
        # Get probe sequence: [t-seq_len+1, ..., t]
        probe_seq = []
        for t_step in range(t - self.seq_len + 1, t + 1):
            probe_step = []
            for p in range(self.num_probes):
                probe_values = []
                for var in self.variables:
                    # Check if probe data exists
                    if p not in self.probe_data or var not in self.probe_data[p]:
                        # Use NaN if probe data is missing
                        probe_values.append(0.0)
                        continue
                    
                    value = self.probe_data[p][var][t_step]
                    # Handle scalar or array
                    if isinstance(value, np.ndarray):
                        value = value.item() if value.size == 1 else value.mean()
                    probe_values.append(float(value))
                probe_step.append(probe_values)
            probe_seq.append(probe_step)
        
        # Convert to tensor: (seq_len, num_probes, num_variables)
        probe_seq = torch.tensor(probe_seq, dtype=torch.float32)
        
        # Get target image at time t
        target_img = []
        for var in self.variables:
            img = self.sim_data[var][t]  # (H, W)
            target_img.append(img)
        
        # Stack to (num_variables, H, W)
        target_img = np.stack(target_img, axis=0)
        target_img = torch.from_numpy(target_img).float()
        
        # Normalize target image using global statistics (per-channel)
        if self.stats is not None:
            for i, var in enumerate(self.variables):
                mean = self.stats[var]['mean']
                std = self.stats[var]['std']
                target_img[i] = (target_img[i] - mean) / (std + 1e-8)
        else:
            # Fallback: per-sample normalization (not recommended)
            target_img = (target_img - target_img.mean()) / (target_img.std() + 1e-8)
        
        # Normalize probe sequence (per variable) using global statistics
        if self.stats is not None:
            for v, var in enumerate(self.variables):
                mean = self.stats[var]['mean']
                std = self.stats[var]['std']
                probe_seq[:, :, v] = (probe_seq[:, :, v] - mean) / (std + 1e-8)
        else:
            # Fallback: per-sequence normalization
            for v in range(len(self.variables)):
                var_values = probe_seq[:, :, v]
                mean = var_values.mean()
                std = var_values.std()
                probe_seq[:, :, v] = (var_values - mean) / (std + 1e-8)
        
        return probe_seq, target_img


def train_epoch(model, dataloader, optimizer, device, kl_weight, scale_weights, use_amp, scaler, vae_model):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_kl_match = 0.0
    num_batches = 0
    
    for probe_seq, target_img in dataloader:
        probe_seq = probe_seq.to(device)  # (B, seq_len, num_probes, num_variables)
        target_img = target_img.to(device)  # (B, C, H, W)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # Get probe-based reconstruction
                outputs, mu_probe, logvar_probe = model(probe_seq)
                
                # Get target latent distribution from VAE encoder
                with torch.no_grad():
                    mu_target, logvar_target = vae_model.encoder(target_img)
                
                # Reconstruction loss
                loss, recon_loss, kl_loss, _ = elbow_loss(
                    outputs, target_img, mu_probe, logvar_probe, kl_weight, scale_weights
                )
                
                # Additional KL divergence to match target latent distribution
                kl_match = 0.5 * torch.sum(
                    logvar_target - logvar_probe + 
                    (torch.exp(logvar_probe) + (mu_probe - mu_target)**2) / torch.exp(logvar_target) - 1
                )
                kl_match = kl_match.mean()
                
                total_loss_batch = loss + 0.1 * kl_match
            
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Get probe-based reconstruction
            outputs, mu_probe, logvar_probe = model(probe_seq)
            
            # Get target latent distribution from VAE encoder
            with torch.no_grad():
                mu_target, logvar_target = vae_model.encoder(target_img)
            
            # Reconstruction loss
            loss, recon_loss, kl_loss, _ = elbow_loss(
                outputs, target_img, mu_probe, logvar_probe, kl_weight, scale_weights
            )
            
            # Additional KL divergence to match target latent distribution
            kl_match = 0.5 * torch.sum(
                logvar_target - logvar_probe + 
                (torch.exp(logvar_probe) + (mu_probe - mu_target)**2) / torch.exp(logvar_target) - 1
            )
            kl_match = kl_match.mean()
            
            total_loss_batch = loss + 0.1 * kl_match
            total_loss_batch.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        total_kl_match += kl_match.item()
        num_batches += 1
    
    return (total_loss / num_batches, total_recon / num_batches, 
            total_kl / num_batches, total_kl_match / num_batches)


@torch.no_grad()
def validate(model, dataloader, device, kl_weight, scale_weights, vae_model):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_kl_match = 0.0
    num_batches = 0
    
    for probe_seq, target_img in dataloader:
        probe_seq = probe_seq.to(device)
        target_img = target_img.to(device)
        
        outputs, mu_probe, logvar_probe = model(probe_seq)
        
        # Get target latent
        mu_target, logvar_target = vae_model.encoder(target_img)
        
        loss, recon_loss, kl_loss, _ = elbow_loss(
            outputs, target_img, mu_probe, logvar_probe, kl_weight, scale_weights
        )
        
        kl_match = 0.5 * torch.sum(
            logvar_target - logvar_probe + 
            (torch.exp(logvar_probe) + (mu_probe - mu_target)**2) / torch.exp(logvar_target) - 1
        )
        kl_match = kl_match.mean()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        total_kl_match += kl_match.item()
        num_batches += 1
    
    return (total_loss / num_batches, total_recon / num_batches, 
            total_kl / num_batches, total_kl_match / num_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", type=str, required=True, 
                       help="Path to trained VAE checkpoint")
    parser.add_argument("--sim-data-dir", type=str, 
                       default="/dtu/blackhole/1b/223803/tcv_probe_data/simulation")
    parser.add_argument("--probe-data-dir", type=str,
                       default="/dtu/blackhole/1b/223803/tcv_probe_data/probes")
    parser.add_argument("--variables", type=str, nargs="+", default=["n", "phi"])
    parser.add_argument("--seq-len", type=int, default=10, 
                       help="Number of past timesteps for probe encoder")
    parser.add_argument("--num-probes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--output", type=str, 
                       default="/dtu/blackhole/1b/223803/runs/probe_reconstruction")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Probe Reconstruction Model")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Simulation data: {args.sim_data_dir}")
    print(f"Probe data: {args.probe_data_dir}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of probes: {args.num_probes}")
    
    # Load frozen VAE
    print("\nLoading pretrained VAE...")
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae_model = MultiScaleVAE(
        in_channels=len(args.variables),
        latent_dim=256,
        base_channels=32,
        target_sizes=[64, 128, 256, 512],
    ).to(device)
    vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    print(f"Loaded VAE from epoch {vae_checkpoint['epoch']}")
    
    # Create probe encoder
    print("\nCreating temporal probe encoder...")
    probe_encoder = TemporalProbeEncoder(
        num_probes=args.num_probes,
        num_variables=len(args.variables),
        seq_len=args.seq_len,
        latent_dim=256,
        hidden_dim=512,
    )
    
    # Create combined model
    model = ProbeVAE(vae_model, probe_encoder).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    print("\nLoading dataset...")
    dataset = ProbeReconstructionDataset(
        sim_data_path=Path(args.sim_data_dir),
        probe_data_path=Path(args.probe_data_dir),
        variables=args.variables,
        seq_len=args.seq_len,
        num_probes=args.num_probes,
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, change to 4 after testing
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging, change to 4 after testing
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler(device='cuda') if args.use_amp else None
    
    scale_weights = {64: 0.5, 128: 1.0, 256: 2.0, 512: 4.0}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_recon, train_kl, train_kl_match = train_epoch(
            model, train_loader, optimizer, device, args.kl_weight, 
            scale_weights, args.use_amp, scaler, vae_model
        )
        
        val_loss, val_recon, val_kl, val_kl_match = validate(
            model, val_loader, device, args.kl_weight, scale_weights, vae_model
        )
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}, KLm={train_kl_match:.4f}) | "
              f"Val: {val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f}, KLm={val_kl_match:.4f}) | {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'probe_encoder_state_dict': probe_encoder.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, output_dir / "best_model.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={args.patience})")
                break
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
