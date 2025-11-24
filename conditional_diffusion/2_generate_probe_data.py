#!/usr/bin/env python3
"""
Generate probe sampling data from extracted BOUT++ arrays.

Samples probe measurements along a horizontal line at y=half with 40 points.
Saves probe coordinates and values for each timestep/frame.
"""

import argparse
from pathlib import Path
import numpy as np
from typing import Tuple


def generate_probe_line(
    num_probes: int,
    x_size: int,
    y_position: int,
) -> np.ndarray:
    """
    Generate probe coordinates along a horizontal line.
    
    Parameters
    ----------
    num_probes : int
        Number of probes to generate
    x_size : int
        Size of the x dimension
    y_position : int
        Y coordinate for the horizontal line
    
    Returns
    -------
    coords : ndarray of shape (num_probes, 2)
        Probe coordinates as (x, y) pairs
    """
    # Evenly spaced probes along x axis
    x_coords = np.linspace(0, x_size - 1, num_probes, dtype=int)
    y_coords = np.full(num_probes, y_position, dtype=int)
    
    coords = np.stack([x_coords, y_coords], axis=1)
    return coords


def sample_probes_from_field(
    field: np.ndarray,
    probe_coords: np.ndarray,
) -> np.ndarray:
    """
    Sample field values at probe locations.
    
    Parameters
    ----------
    field : ndarray of shape (time, x, z) or (x, z)
        Field data to sample from
    probe_coords : ndarray of shape (num_probes, 2)
        Probe coordinates as (x, z) pairs
    
    Returns
    -------
    values : ndarray
        Sampled values at probe locations
        Shape: (time, num_probes) or (num_probes,)
    """
    x_indices = probe_coords[:, 0]
    z_indices = probe_coords[:, 1]
    
    if field.ndim == 3:
        # (time, x, z) -> (time, num_probes)
        values = field[:, x_indices, z_indices]
    elif field.ndim == 2:
        # (x, z) -> (num_probes,)
        values = field[x_indices, z_indices]
    else:
        raise ValueError(f"Unexpected field shape: {field.shape}")
    
    return values


def generate_probe_data(
    data_dir: str | Path,
    output_dir: str | Path,
    variables: list[str] = None,
    num_probes: int = 40,
):
    """
    Generate probe sampling data from extracted BOUT++ arrays.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing extracted .npy files
    output_dir : str or Path
        Directory to save probe data
    variables : list of str, optional
        Variables to process (default: ['n', 'te', 'ti', 'phi'])
    num_probes : int, optional
        Number of probes to sample (default: 40)
    """
    if variables is None:
        variables = ['n', 'te', 'ti', 'phi']
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {data_dir}")
    print(f"Number of probes: {num_probes}")
    
    # Load first variable to get dimensions
    first_var = variables[0]
    first_data = np.load(data_dir / f"{first_var}.npy")
    print(f"\nData shape ({first_var}): {first_data.shape}")
    
    # Determine spatial dimensions
    if first_data.ndim == 3:
        time_steps, x_size, z_size = first_data.shape
    else:
        raise ValueError(f"Expected 3D data (time, x, z), got shape {first_data.shape}")
    
    # Generate probe coordinates at y=half (middle of z dimension)
    y_position = z_size // 2
    probe_coords = generate_probe_line(num_probes, x_size, y_position)
    
    print(f"Probe line at z={y_position} (middle of {z_size})")
    print(f"Probe x-coordinates: {probe_coords[:, 0]}")
    print(f"Probe coordinates shape: {probe_coords.shape}")
    
    # Save probe coordinates
    coords_path = output_dir / "probe_coordinates.npy"
    np.save(coords_path, probe_coords)
    print(f"\nSaved probe coordinates to: {coords_path}")
    
    # Process each variable
    for var_name in variables:
        print(f"\nProcessing variable: {var_name}")
        
        # Load data
        data_path = data_dir / f"{var_name}.npy"
        if not data_path.exists():
            print(f"  WARNING: {data_path} not found, skipping")
            continue
        
        data = np.load(data_path)
        print(f"  Data shape: {data.shape}")
        
        # Sample at probe locations
        probe_values = sample_probes_from_field(data, probe_coords)
        print(f"  Probe values shape: {probe_values.shape}")
        print(f"  Value range: [{probe_values.min():.3e}, {probe_values.max():.3e}]")
        
        # Save probe values
        output_path = output_dir / f"probe_{var_name}.npy"
        np.save(output_path, probe_values)
        print(f"  Saved to: {output_path}")
    
    print(f"\nProbe generation complete! Data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate probe sampling data from extracted BOUT++ arrays"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/bout_data",
        help="Directory containing extracted .npy files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/probe_data",
        help="Output directory for probe data"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "te", "ti", "phi"],
        help="Variables to process"
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=40,
        help="Number of probes to sample"
    )
    
    args = parser.parse_args()
    
    generate_probe_data(
        data_dir=args.data_dir,
        output_dir=args.output,
        variables=args.variables,
        num_probes=args.num_probes,
    )


if __name__ == "__main__":
    main()
