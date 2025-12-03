#!/usr/bin/env python3
"""
Dataset for loading TCV data extracted from BOUT++ files.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
import json


class TCVDataset(Dataset):
    """
    Dataset for TCV simulation data.
    
    Loads data with shape (501, 514, 1, 512) and reshapes to (501, 512, 512)
    by removing dimensions at index 1 and 2.
    """
    
    def __init__(self, data_dir, variables=None):
        """
        Parameters
        ----------
        data_dir : str or Path
            Directory containing variable .npy files
        variables : list of str, optional
            Variables to load (default: ['n', 'phi'])
        """
        if variables is None:
            variables = ['n', 'phi']
        
        self.data_dir = Path(data_dir)
        self.variables = variables
        
        # Load global statistics for normalization
        stats_path = self.data_dir / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded global statistics from {stats_path}")
        else:
            print(f"WARNING: No statistics file found at {stats_path}")
            print("Will compute per-sample normalization (not recommended)")
            self.stats = None
        
        # Load all variables
        self.data_list = []
        for var_name in variables:
            var_path = self.data_dir / f"{var_name}.npy"
            if not var_path.exists():
                raise FileNotFoundError(f"Variable file not found: {var_path}")
            
            print(f"Loading {var_name} from {var_path}")
            data = np.load(var_path, mmap_mode='r')  # Memory-mapped for large files
            
            print(f"  Original shape: {data.shape}")
            
            # Reshape from (501, 514, 1, 512) to (501, 512, 512)
            # Remove dimension at index 1 (514 -> skip first 2 entries)
            # Remove dimension at index 2 (1 -> squeeze)
            if len(data.shape) == 4:
                # Assume shape is (T, X, 1, Y)
                # Take X from index 2 onwards (skip first 2)
                data = data[:, 2:, 0, :]  # (501, 512, 512)
            elif len(data.shape) == 3:
                # If already (T, X, Y), just skip first 2 in X
                data = data[:, 2:, :]
            
            print(f"  Reshaped to: {data.shape}")
            self.data_list.append(data)
        
        # Stack variables along channel dimension
        # Each variable: (T, H, W) -> stacked: (T, C, H, W)
        self.num_samples = self.data_list[0].shape[0]
        
        print(f"Dataset initialized with {self.num_samples} samples")
        print(f"Variables: {variables}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns
        -------
        data : torch.Tensor
            Shape: (C, H, W) where C = number of variables
        """
        # Stack all variables for this timestep
        sample_list = []
        for var_data in self.data_list:
            sample_list.append(var_data[idx])  # (H, W)
        
        # Stack to (C, H, W)
        sample = np.stack(sample_list, axis=0)
        
        # Convert to tensor and ensure float32
        sample = torch.from_numpy(sample).float()
        
        # Normalize using global statistics (per-channel)
        if self.stats is not None:
            for i, var in enumerate(self.variables):
                mean = self.stats[var]['mean']
                std = self.stats[var]['std']
                sample[i] = (sample[i] - mean) / (std + 1e-8)
        else:
            # Fallback: per-sample normalization (not recommended)
            sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        
        return sample


def build_dataloaders(
    data_dir,
    variables=None,
    batch_size=8,
    num_workers=4,
    train_split=0.8,
    **kwargs  # Accept and ignore other arguments for compatibility
):
    """
    Build train and validation dataloaders.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing variable .npy files
    variables : list of str, optional
        Variables to load
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
    train_split : float
        Fraction of data to use for training
    
    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    dataset = TCVDataset(data_dir, variables)
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
    # Test the dataset
    data_dir = "/dtu/blackhole/1b/223803/tcv_data"
    
    print("Testing TCVDataset...")
    dataset = TCVDataset(data_dir, variables=["n", "phi"])
    
    print(f"\nDataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    print("\nTesting dataloaders...")
    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        variables=["n", "phi"],
        batch_size=4,
        num_workers=0,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}")
