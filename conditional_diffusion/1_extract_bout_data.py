#!/usr/bin/env python3
"""
Extract BOUT++ simulation data and save as numpy arrays.

Reads n, te, ti, phi from BOUT++ NetCDF dump files and saves them
as .npy files for downstream processing.
"""

import argparse
from pathlib import Path
import numpy as np
from boutdata import collect


def extract_bout_data(
    bout_path: str | Path,
    output_dir: str | Path,
    variables: list[str] = None,
    trim_radial: int = 2,
    squeeze_axis: int = 2,
):
    """
    Extract BOUT++ data and save as numpy arrays.
    
    Parameters
    ----------
    bout_path : str or Path
        Path to BOUT++ data directory containing BOUT.dmp.*.nc files
    output_dir : str or Path
        Directory to save extracted numpy arrays
    variables : list of str, optional
        Variables to extract (default: ['n', 'te', 'ti', 'phi'])
    trim_radial : int, optional
        Number of radial guard cells to trim from edges (default: 2)
    squeeze_axis : int, optional
        Axis to squeeze out (default: 2)
    """
    if variables is None:
        variables = ['n', 'te', 'ti', 'phi']
    
    bout_path = Path(bout_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading BOUT++ data from: {bout_path}")
    print(f"Variables to extract: {variables}")
    
    for var_name in variables:
        print(f"\nProcessing variable: {var_name}")
        
        # Collect data from BOUT++ files
        data = collect(var_name, path=str(bout_path), strict=False)
        print(f"  Raw shape: {data.shape}")
        
        # Squeeze singleton dimensions
        if squeeze_axis is not None and data.shape[squeeze_axis] == 1:
            data = np.squeeze(data, axis=squeeze_axis)
            print(f"  After squeeze: {data.shape}")
        
        # Trim radial guard cells
        if trim_radial > 0:
            # Assumes radial dimension is second-to-last: (time, x, z) or (time, x, y, z)
            data = data[..., trim_radial:-trim_radial, :]
            print(f"  After trim: {data.shape}")
        
        # Save as numpy array
        output_path = output_dir / f"{var_name}.npy"
        np.save(output_path, data)
        print(f"  Saved to: {output_path}")
        print(f"  Final shape: {data.shape}")
        print(f"  Data range: [{data.min():.3e}, {data.max():.3e}]")
    
    print(f"\nExtraction complete! Data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract BOUT++ data to numpy arrays"
    )
    parser.add_argument(
        "--bout-path",
        type=str,
        default="/dtu/blackhole/1b/223803/Data_Files/Data_Files",
        help="Path to BOUT++ data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/bout_data",
        help="Output directory for numpy arrays"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "te", "ti", "phi"],
        help="Variables to extract"
    )
    parser.add_argument(
        "--trim-radial",
        type=int,
        default=2,
        help="Number of radial guard cells to trim"
    )
    parser.add_argument(
        "--squeeze-axis",
        type=int,
        default=2,
        help="Axis to squeeze out (set to -1 to disable)"
    )
    
    args = parser.parse_args()
    
    squeeze_axis = None if args.squeeze_axis < 0 else args.squeeze_axis
    
    extract_bout_data(
        bout_path=args.bout_path,
        output_dir=args.output,
        variables=args.variables,
        trim_radial=args.trim_radial,
        squeeze_axis=squeeze_axis,
    )


if __name__ == "__main__":
    main()
