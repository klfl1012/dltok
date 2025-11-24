#!/usr/bin/env python3
"""
Dataset for VAE denoising training.

Loads extracted BOUT++ numpy arrays and creates training pairs of
(clean image, noisy image) for denoising VAE training.
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


class DenoisingDataset(Dataset):
    """
    Dataset that loads tokamak field data and adds noise for denoising training.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing extracted .npy files
    variables : list of str
        Variables to load (e.g., ['n', 'te', 'ti', 'phi'])
    noise_level : float or tuple of float
        Standard deviation of Gaussian noise to add (relative to data std).
        If tuple (min, max), random noise level sampled per batch.
    train : bool
        If True, use train split; otherwise use validation split
    train_fraction : float
        Fraction of data to use for training
    normalize : bool
        If True, normalize each variable to zero mean and unit std
    resize : int or None
        If provided, resize images to (resize, resize) using bilinear interpolation
    lazy_load : bool
        If True, load data on-demand instead of caching in memory (saves RAM)
    probe_augmentation_prob : float
        Probability of using probe-based interpolation instead of noise (default: 0.4)
    num_probes : int
        Number of random probe points for interpolation augmentation (default: 40)
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        variables: List[str] = None,
        noise_level: float | Tuple[float, float] = 0.1,
        train: bool = True,
        train_fraction: float = 0.8,
        normalize: bool = True,
        resize: int = None,
        lazy_load: bool = False,
        probe_augmentation_prob: float = 0.4,
        num_probes: int = 40,
    ):
        if variables is None:
            variables = ['n', 'te', 'ti', 'phi']
        
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.train = train
        self.normalize = normalize
        self.resize = resize
        self.lazy_load = lazy_load
        self.probe_augmentation_prob = probe_augmentation_prob
        self.num_probes = num_probes
        
        # Handle noise level (single value or range)
        if isinstance(noise_level, (tuple, list)):
            self.noise_min, self.noise_max = noise_level
            self.multi_noise = True
        else:
            self.noise_level = noise_level
            self.multi_noise = False
        
        # Compute statistics and optionally load data
        self.data = {} if not lazy_load else None
        self.stats = {}
        self.file_paths = {}
        
        print(f"Loading data from {self.data_dir}")
        print(f"Lazy load: {lazy_load}, Resize: {resize if resize else 'None'}")
        print(f"Noise: {'random' if self.multi_noise else self.noise_level}")
        print(f"Probe augmentation: {probe_augmentation_prob*100:.0f}% with {num_probes} probes")
        
        for var_name in self.variables:
            var_path = self.data_dir / f"{var_name}.npy"
            if not var_path.exists():
                raise FileNotFoundError(f"Variable file not found: {var_path}")
            
            self.file_paths[var_name] = var_path
            
            # Load to compute stats
            data = np.load(var_path)  # Shape: (time, x, z)
            print(f"  {var_name}: {data.shape}, range [{data.min():.3e}, {data.max():.3e}]")
            
            # Compute statistics for normalization
            mean = data.mean()
            std = data.std()
            self.stats[var_name] = {'mean': mean, 'std': std}
            
            # Normalize if requested
            if self.normalize:
                data = (data - mean) / (std + 1e-8)
                print(f"    Normalized: mean={data.mean():.3e}, std={data.std():.3e}")
            
            # Store in memory if not lazy loading
            if not lazy_load:
                self.data[var_name] = data
            
            # Store total samples
            if var_name == self.variables[0]:
                total_samples = data.shape[0]
        
        # Create train/val split
        split_idx = int(total_samples * train_fraction)
        
        if self.train:
            self.indices = np.arange(0, split_idx)
        else:
            self.indices = np.arange(split_idx, total_samples)
        
        mode = 'train' if train else 'val'
        print(f"Split: {mode}, {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _create_probe_interpolated(self, clean: torch.Tensor) -> torch.Tensor:
        """
        Create noisy input by sampling random points and interpolating.
        
        Parameters
        ----------
        clean : torch.Tensor
            Clean image, shape (num_vars, H, W)
        
        Returns
        -------
        noisy : torch.Tensor
            Interpolated image from random probe points
        """
        num_vars, H, W = clean.shape
        
        # Generate random probe positions on a horizontal line
        # Random z position for the horizontal line
        probe_z = np.random.randint(0, H)
        
        # Generate random x positions for probes (sorted)
        probe_x = np.sort(np.random.choice(W, size=min(self.num_probes, W), replace=False))
        
        # Create interpolated image for each variable
        noisy_channels = []
        for var_idx in range(num_vars):
            clean_channel = clean[var_idx].numpy()  # (H, W)
            
            # Sample values at probe locations
            probe_values = clean_channel[probe_z, probe_x]
            
            # Create 1D interpolator
            interpolator = interp1d(
                probe_x,
                probe_values,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            # Interpolate along x-axis
            x_coords = np.arange(W)
            interpolated_line = interpolator(x_coords)
            
            # Create image with exponential decay from probe line
            noisy_channel = np.zeros((H, W))
            for z in range(H):
                distance = abs(z - probe_z)
                decay = np.exp(-distance / (H / 10))  # Decay over ~10% of height
                noisy_channel[z, :] = interpolated_line * decay
            
            noisy_channels.append(noisy_channel)
        
        noisy = torch.from_numpy(np.stack(noisy_channels, axis=0)).float()
        return noisy
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a pair of (clean, noisy) images.
        
        Returns
        -------
        clean : torch.Tensor
            Clean image, shape (num_vars, H, W) or (num_vars, resize, resize)
        noisy : torch.Tensor
            Noisy version, shape (num_vars, H, W) or (num_vars, resize, resize)
        """
        time_idx = self.indices[idx]
        
        # Stack all variables into single tensor
        frames = []
        for var_name in self.variables:
            if self.lazy_load:
                # Load on-demand
                data = np.load(self.file_paths[var_name])
                frame = data[time_idx]
                
                # Apply normalization
                if self.normalize:
                    mean = self.stats[var_name]['mean']
                    std = self.stats[var_name]['std']
                    frame = (frame - mean) / (std + 1e-8)
            else:
                # Use cached data
                frame = self.data[var_name][time_idx]  # (H, W)
            
            frames.append(frame)
        
        clean = np.stack(frames, axis=0)  # (num_vars, H, W)
        
        # Convert to torch tensor first
        clean = torch.from_numpy(clean).float()
        
        # Resize if requested
        if self.resize is not None:
            clean = F.interpolate(
                clean.unsqueeze(0),  # Add batch dim
                size=(self.resize, self.resize),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim
        
        # Decide whether to use probe interpolation or Gaussian noise
        use_probe = np.random.random() < self.probe_augmentation_prob
        
        if use_probe:
            # Use probe-based interpolation
            noisy = self._create_probe_interpolated(clean)
        else:
            # Sample noise level if multi-noise
            if self.multi_noise:
                noise_std = np.random.uniform(self.noise_min, self.noise_max)
            else:
                noise_std = self.noise_level
            
            # Add Gaussian noise
            noise = torch.randn_like(clean) * noise_std
            noisy = clean + noise
        
        return clean, noisy
    
    def get_stats(self) -> dict:
        """Return normalization statistics for each variable."""
        return self.stats


def build_dataloaders(
    data_dir: str | Path,
    variables: List[str] = None,
    noise_level: float | Tuple[float, float] = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    train_fraction: float = 0.8,
    normalize: bool = True,
    pin_memory: bool = True,
    resize: int = None,
    lazy_load: bool = False,
    probe_augmentation_prob: float = 0.4,
    num_probes: int = 40,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing extracted .npy files
    variables : list of str, optional
        Variables to load
    noise_level : float or tuple of (min, max)
        Noise standard deviation. If tuple, random level per batch.
    batch_size : int
        Batch size for training
    num_workers : int
        Number of dataloader workers
    train_fraction : float
        Fraction of data for training
    normalize : bool
        Whether to normalize data
    pin_memory : bool
        Whether to pin memory for faster GPU transfer
    resize : int, optional
        If provided, resize images to (resize, resize)
    lazy_load : bool
        If True, load data on-demand instead of caching (saves RAM)
    probe_augmentation_prob : float
        Probability of using probe-based interpolation (default: 0.4)
    num_probes : int
        Number of random probe points for interpolation (default: 40)
    
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    if variables is None:
        variables = ['n', 'te', 'ti', 'phi']
    
    train_dataset = DenoisingDataset(
        data_dir=data_dir,
        variables=variables,
        noise_level=noise_level,
        train=True,
        train_fraction=train_fraction,
        normalize=normalize,
        resize=resize,
        lazy_load=lazy_load,
        probe_augmentation_prob=probe_augmentation_prob,
        num_probes=num_probes,
    )
    
    val_dataset = DenoisingDataset(
        data_dir=data_dir,
        variables=variables,
        noise_level=noise_level,
        train=False,
        train_fraction=train_fraction,
        normalize=normalize,
        resize=resize,
        lazy_load=lazy_load,
        probe_augmentation_prob=probe_augmentation_prob,
        num_probes=num_probes,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """Test dataset loading."""
    parser = argparse.ArgumentParser(description="Test denoising dataset")
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
        help="Variables to load"
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
        help="Max noise level for random sampling"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
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
        help="Load data on-demand instead of caching in memory"
    )
    
    args = parser.parse_args()
    
    # Handle noise level range
    if args.noise_max is not None:
        noise_level = (args.noise_level, args.noise_max)
    else:
        noise_level = args.noise_level
    
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        variables=args.variables,
        noise_level=noise_level,
        batch_size=args.batch_size,
        resize=args.resize,
        lazy_load=args.lazy_load,
    )
    
    # Test loading a batch
    print("\nTesting batch loading...")
    clean, noisy = next(iter(train_loader))
    print(f"Clean batch shape: {clean.shape}")
    print(f"Noisy batch shape: {noisy.shape}")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # Print stats
    stats = train_loader.dataset.get_stats()
    print("\nNormalization statistics:")
    for var_name, var_stats in stats.items():
        print(f"  {var_name}: mean={var_stats['mean']:.3e}, std={var_stats['std']:.3e}")


if __name__ == "__main__":
    main()
