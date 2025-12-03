#!/usr/bin/env python3
"""
Calculate global mean and std for the entire dataset.
"""

import numpy as np
from pathlib import Path
import json

data_dir = Path("/dtu/blackhole/1b/223803/tcv_data")
variables = ["n", "phi"]

print("Calculating global dataset statistics...")
print("=" * 60)

stats = {}

for var in variables:
    var_path = data_dir / f"{var}.npy"
    print(f"\nProcessing {var}...")
    
    # Load with memory mapping
    data = np.load(var_path, mmap_mode='r')
    print(f"  Original shape: {data.shape}")
    
    # Apply same reshaping as in dataset
    if len(data.shape) == 4:
        data = data[:, 2:, 0, :]  # (T, 512, 512)
    elif len(data.shape) == 3:
        data = data[:, 2:, :]
    
    print(f"  Reshaped to: {data.shape}")
    
    # Calculate statistics
    # Use float64 for precision in accumulation
    mean = np.mean(data.astype(np.float64))
    std = np.std(data.astype(np.float64))
    min_val = np.min(data)
    max_val = np.max(data)
    
    stats[var] = {
        'mean': float(mean),
        'std': float(std),
        'min': float(min_val),
        'max': float(max_val),
    }
    
    print(f"  Mean:  {mean:.6e}")
    print(f"  Std:   {std:.6e}")
    print(f"  Min:   {min_val:.6e}")
    print(f"  Max:   {max_val:.6e}")
    print(f"  Range: [{min_val:.6e}, {max_val:.6e}]")

# Save statistics
stats_path = data_dir / "dataset_stats.json"
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print(f"Statistics saved to: {stats_path}")
print("\nTo use in dataset:")
print("  Load this JSON file and use these global mean/std values")
print("  instead of computing per-sample statistics.")
