#!/usr/bin/env python3
"""
Extract TCV data from BOUT++ files.

Extracts both simulation and probe data from BOUT++ NetCDF files
and saves to blackhole storage as numpy arrays.
"""

import argparse
from pathlib import Path
import numpy as np
from boutdata import collect


def extract_tcv_data(
    bout_path: str | Path,
    output_dir: str | Path,
    variables: list[str] = None,
):
    """
    Extract TCV data from BOUT++ files and save as numpy arrays.
    
    Parameters
    ----------
    bout_path : str or Path
        Path to BOUT++ data directory
    output_dir : str or Path
        Directory to save extracted numpy arrays
    variables : list of str, optional
        Variables to extract (default: ['n', 'phi'])
    """
    if variables is None:
        variables = ['n', 'phi']
    
    bout_path = Path(bout_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TCV DATA EXTRACTION FROM BOUT++")
    print("=" * 70)
    print(f"BOUT++ data path: {bout_path}")
    print(f"Variables to extract: {variables}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print("=" * 70)
    
    for var_name in variables:
        print(f"\nExtracting data for: {var_name}")
        
        try:
            # Collect from BOUT++ files
            print(f"  Reading from BOUT++ path: {bout_path}")
            data = collect(var_name, path=str(bout_path), strict=False)
            
            if data.size == 0:
                print(f"  WARNING: Collected data is EMPTY!")
                continue
            
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Data range: [{data.min():.3e}, {data.max():.3e}]")
            print(f"  Mean: {data.mean():.3e}, Std: {data.std():.3e}")
            
            # Save to output
            output_path = output_dir / f"{var_name}.npy"
            np.save(output_path, data)
            print(f"  ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"  ERROR extracting data: {e}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract TCV data from BOUT++ files"
    )
    parser.add_argument(
        "--bout-path",
        type=str,
        default="/dtu-compute/proj-jehi/TCV-DATA",
        help="Path to BOUT++ data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/tcv_data",
        help="Output directory for numpy arrays (on blackhole)"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "phi"],
        help="Variables to extract"
    )
    
    args = parser.parse_args()
    
    extract_tcv_data(
        bout_path=args.bout_path,
        output_dir=args.output,
        variables=args.variables,
    )


if __name__ == "__main__":
    main()
