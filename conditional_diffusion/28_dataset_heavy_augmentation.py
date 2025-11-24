#!/usr/bin/env python3
"""
Enhanced dataset with heavy probe-based augmentation.

Generates 5x more training samples per image by creating multiple
probe-based interpolation variants.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata


class HeavyAugmentationDataset(Dataset):
    """
    Dataset with heavy probe-based augmentation.
    
    For each ground truth image, generates multiple training samples using:
    - Random probe positions
    - Different interpolation methods
    - Varying noise levels
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing .npy files
    variables : list of str
        Variables to load
    noise_level : float or tuple
        Noise standard deviation (or (min, max) for random range)
    train : bool
        If True, use train split; otherwise validation split
    train_fraction : float
        Fraction of data for training
    resize : int, optional
        Resize images to this size
    lazy_load : bool
        Load data on-demand to save memory
    samples_per_image : int
        Number of augmented samples to generate per image (default: 5)
    num_probes_range : tuple
        Range of number of probes to sample (min, max)
    probe_noise_std : float
        Noise to add to probe values
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        variables: List[str] = None,
        noise_level: Union[float, Tuple[float, float]] = 0.1,
        train: bool = True,
        train_fraction: float = 0.8,
        resize: int = None,
        lazy_load: bool = False,
        samples_per_image: int = 5,
        num_probes_range: Tuple[int, int] = (20, 60),
        probe_noise_std: float = 0.05,
    ):
        if variables is None:
            variables = ['n', 'te', 'ti', 'phi']
        
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.noise_level = noise_level
        self.train = train
        self.resize = resize
        self.lazy_load = lazy_load
        self.samples_per_image = samples_per_image
        self.num_probes_range = num_probes_range
        self.probe_noise_std = probe_noise_std
        
        # Determine if noise is a range
        if isinstance(noise_level, (list, tuple)):
            self.noise_min, self.noise_max = noise_level
            self.noise_is_range = True
        else:
            self.noise_min = noise_level
            self.noise_max = noise_level
            self.noise_is_range = False
        
        # Load or store paths
        self.data = {}
        self.stats = {}
        
        print(f"Loading {'training' if train else 'validation'} data...")
        print(f"Samples per image: {samples_per_image} (total: {samples_per_image}x original)")
        print(f"Probe range: {num_probes_range[0]}-{num_probes_range[1]} probes")
        
        for var_name in self.variables:
            var_path = self.data_dir / f"{var_name}.npy"
            if not var_path.exists():
                raise FileNotFoundError(f"Variable file not found: {var_path}")
            
            if lazy_load:
                self.data[var_name] = var_path
            else:
                self.data[var_name] = np.load(var_path)
            
            # Compute stats from full data
            full_data = np.load(var_path) if lazy_load else self.data[var_name]
            self.stats[var_name] = {
                'mean': full_data.mean(),
                'std': full_data.std()
            }
            
            print(f"  {var_name}: shape={full_data.shape}, mean={self.stats[var_name]['mean']:.4f}, std={self.stats[var_name]['std']:.4f}")
            
            if lazy_load:
                del full_data
        
        # Determine train/val split
        sample_data = np.load(self.data[self.variables[0]]) if lazy_load else self.data[self.variables[0]]
        total_frames = sample_data.shape[0]
        split_idx = int(total_frames * train_fraction)
        
        if train:
            self.indices = np.arange(0, split_idx)
        else:
            self.indices = np.arange(split_idx, total_frames)
        
        print(f"Total frames: {len(self.indices)} (from {total_frames} total)")
        print(f"Effective dataset size: {len(self.indices) * samples_per_image}")
    
    def __len__(self):
        return len(self.indices) * self.samples_per_image
    
    def _load_frame(self, var_name: str, idx: int) -> np.ndarray:
        """Load a single frame."""
        if self.lazy_load:
            data = np.load(self.data[var_name])
            frame = data[idx]
        else:
            frame = self.data[var_name][idx]
        return frame
    
    def _create_probe_augmented_input(
        self,
        clean_image: np.ndarray,
        num_probes: int,
    ) -> np.ndarray:
        """
        Create augmented input from random probe sampling.
        
        Parameters
        ----------
        clean_image : np.ndarray
            Ground truth image, shape (H, W)
        num_probes : int
            Number of random probe points
        
        Returns
        -------
        aug_image : np.ndarray
            Augmented image from probe interpolation, shape (H, W)
        """
        H, W = clean_image.shape
        
        # Sample random probe locations
        probe_x = np.random.randint(0, W, size=num_probes)
        probe_z = np.random.randint(0, H, size=num_probes)
        probe_coords = np.stack([probe_x, probe_z], axis=1)
        
        # Get values at probe locations
        probe_values = clean_image[probe_z, probe_x]
        
        # Add noise to probe values
        if self.probe_noise_std > 0:
            probe_values = probe_values + np.random.randn(num_probes) * self.probe_noise_std
        
        # Interpolate to full image
        grid_x, grid_z = np.meshgrid(np.arange(W), np.arange(H))
        
        aug_image = griddata(
            probe_coords,
            probe_values,
            (grid_x, grid_z),
            method='linear',
            fill_value=clean_image.mean()
        )
        
        # Fill NaN with nearest neighbor
        if np.isnan(aug_image).any():
            mask = np.isnan(aug_image)
            aug_image_nn = griddata(
                probe_coords,
                probe_values,
                (grid_x, grid_z),
                method='nearest'
            )
            aug_image[mask] = aug_image_nn[mask]
        
        return aug_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item with heavy augmentation.
        
        Returns
        -------
        clean : torch.Tensor
            Clean target image, shape (C, H, W)
        noisy : torch.Tensor
            Noisy/augmented input, shape (C, H, W)
        """
        # Map idx to base image and augmentation variant
        base_idx = idx // self.samples_per_image
        aug_variant = idx % self.samples_per_image
        
        frame_idx = self.indices[base_idx]
        
        # Load all variables for this frame
        clean_channels = []
        noisy_channels = []
        
        for var_name in self.variables:
            # Load ground truth
            frame = self._load_frame(var_name, frame_idx)
            
            # Normalize
            mean = self.stats[var_name]['mean']
            std = self.stats[var_name]['std']
            frame_norm = (frame - mean) / (std + 1e-8)
            
            clean_channels.append(frame_norm)
            
            # Create augmented input based on variant
            if aug_variant < self.samples_per_image - 1:
                # Probe-based augmentation
                num_probes = np.random.randint(self.num_probes_range[0], self.num_probes_range[1] + 1)
                aug_frame = self._create_probe_augmented_input(frame_norm, num_probes)
            else:
                # Last variant: standard Gaussian noise
                aug_frame = frame_norm.copy()
            
            # Add noise
            if self.noise_is_range:
                noise_std = np.random.uniform(self.noise_min, self.noise_max)
            else:
                noise_std = self.noise_min
            
            noise = np.random.randn(*aug_frame.shape) * noise_std
            noisy_frame = aug_frame + noise
            
            noisy_channels.append(noisy_frame)
        
        # Stack channels
        clean = np.stack(clean_channels, axis=0).astype(np.float32)
        noisy = np.stack(noisy_channels, axis=0).astype(np.float32)
        
        # Convert to tensors
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        
        # Resize if needed
        if self.resize is not None:
            clean = torch.nn.functional.interpolate(
                clean.unsqueeze(0),
                size=(self.resize, self.resize),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            noisy = torch.nn.functional.interpolate(
                noisy.unsqueeze(0),
                size=(self.resize, self.resize),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return clean, noisy


def build_dataloaders(
    data_dir: str | Path,
    variables: List[str] = None,
    noise_level: Union[float, Tuple[float, float]] = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    resize: int = None,
    lazy_load: bool = False,
    samples_per_image: int = 5,
    num_probes_range: Tuple[int, int] = (20, 60),
    probe_noise_std: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders with heavy augmentation.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory with .npy files
    variables : list of str
        Variables to load
    noise_level : float or tuple
        Noise level(s)
    batch_size : int
        Batch size
    num_workers : int
        Number of dataloader workers
    resize : int, optional
        Resize images
    lazy_load : bool
        Lazy loading
    samples_per_image : int
        Augmentation multiplier
    num_probes_range : tuple
        Range of probe counts
    probe_noise_std : float
        Noise on probe values
    
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    if variables is None:
        variables = ['n', 'te', 'ti', 'phi']
    
    train_dataset = HeavyAugmentationDataset(
        data_dir=data_dir,
        variables=variables,
        noise_level=noise_level,
        train=True,
        resize=resize,
        lazy_load=lazy_load,
        samples_per_image=samples_per_image,
        num_probes_range=num_probes_range,
        probe_noise_std=probe_noise_std,
    )
    
    val_dataset = HeavyAugmentationDataset(
        data_dir=data_dir,
        variables=variables,
        noise_level=noise_level,
        train=False,
        resize=resize,
        lazy_load=lazy_load,
        samples_per_image=1,  # No augmentation for validation
        num_probes_range=num_probes_range,
        probe_noise_std=0.0,  # No probe noise for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing Heavy Augmentation Dataset...")
    
    train_loader, val_loader = build_dataloaders(
        data_dir="/dtu/blackhole/1b/223803/bout_data",
        variables=["n", "te", "ti", "phi"],
        noise_level=(0.05, 0.5),
        batch_size=2,
        num_workers=0,
        resize=256,
        lazy_load=True,
        samples_per_image=5,
        num_probes_range=(20, 60),
        probe_noise_std=0.05,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    clean, noisy = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Clean: {clean.shape}")
    print(f"  Noisy: {noisy.shape}")
    print(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"  Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    print("\n✓ Dataset test passed!")
