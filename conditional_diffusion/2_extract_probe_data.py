#!/usr/bin/env python3
"""
Extract probe data from TCV BOUT++ files.

Reads probe measurements from BOUT.fast.* files and saves them as numpy arrays.
Also extracts probe positions and creates visualization.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
from boutdata import collect


def extract_probe_data(
    bout_path: str | Path,
    output_dir: str | Path,
    field_list: list[str] = None,
    num_probes: int = 64,
):
    """
    Extract probe data from BOUT++ files.
    
    Parameters
    ----------
    bout_path : str or Path
        Path to BOUT++ data directory containing BOUT.fast.* files
    output_dir : str or Path
        Directory to save extracted probe data
    field_list : list of str, optional
        Field variables to extract (default: ['n', 'phi'])
    num_probes : int, optional
        Number of probes (default: 64, from probe 0 to 63)
    """
    if field_list is None:
        field_list = ['n', 'phi']
    
    bout_path = Path(bout_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TCV PROBE DATA EXTRACTION")
    print("=" * 70)
    print(f"BOUT++ data path: {bout_path}")
    print(f"Fields to extract: {field_list}")
    print(f"Number of probes: {num_probes} (0 to {num_probes-1})")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Get grid dimensions for probe positions
    print("\nReading grid dimensions...")
    nx = collect('nx', path=str(bout_path))
    nz = collect('nz', path=str(bout_path))
    print(f"  nx = {nx}, nz = {nz}")
    
    # Extract probe data
    probe_list = [str(i) for i in range(num_probes)]
    probe_data = {}
    probe_pos = {}
    
    print("\n" + "=" * 70)
    print("EXTRACTING PROBE DATA")
    print("=" * 70)
    
    for p in probe_list:
        print(f"\nProbe {p}:")
        probe_data[p] = {}
        probe_pos[p] = {}
        
        # Calculate probe position
        frac = 0.1 * (int(p) + 2)
        probe_pos[p]['x'] = round(frac * int(nx))
        probe_pos[p]['z'] = round(0.5 * int(nz))
        print(f"  Position: x={probe_pos[p]['x']}, z={probe_pos[p]['z']}")
        
        # Build list of variables to extract for this probe
        probe_vars = [f"{f}{p}" for f in field_list]
        probe_vars.append('t_array')
        
        # Extract each variable
        for var in probe_vars:
            found = False
            for file in glob.glob(f'{bout_path}/BOUT.fast.*'):
                try:
                    with Dataset(file) as f:
                        data = f[var][:]
                        # Convert MaskedArray to regular numpy array
                        if hasattr(data, 'filled'):
                            data = data.filled(fill_value=np.nan)
                        probe_data[p][var] = np.array(data)
                        print(f"  ✓ {var}: shape={probe_data[p][var].shape}, loaded from {Path(file).name}")
                        found = True
                        break
                except:
                    continue
            
            if not found:
                print(f"  ✗ WARNING: Variable {var} not found in any BOUT.fast.* file")
    
    # Extract simulation data for reference
    print("\n" + "=" * 70)
    print("EXTRACTING SIMULATION DATA (for reference)")
    print("=" * 70)
    
    sim_data = {}
    sim_data['t_array'] = collect('t_array', path=str(bout_path))
    # Convert MaskedArray to regular array
    if hasattr(sim_data['t_array'], 'filled'):
        sim_data['t_array'] = sim_data['t_array'].filled(fill_value=np.nan)
    sim_data['t_array'] = np.array(sim_data['t_array'])
    print(f"t_array: shape={sim_data['t_array'].shape}")
    
    for f in field_list:
        sim_data[f] = collect(f, path=str(bout_path))
        # Convert MaskedArray to regular array
        if hasattr(sim_data[f], 'filled'):
            sim_data[f] = sim_data[f].filled(fill_value=np.nan)
        sim_data[f] = np.array(sim_data[f])
        print(f"{f}: shape={sim_data[f].shape}")
    
    # Save probe data
    print("\n" + "=" * 70)
    print("SAVING PROBE DATA")
    print("=" * 70)
    
    probe_data_dir = output_dir / "probes"
    probe_data_dir.mkdir(exist_ok=True)
    
    for p in probe_list:
        probe_dir = probe_data_dir / f"probe_{p}"
        probe_dir.mkdir(exist_ok=True)
        
        for var, data in probe_data[p].items():
            output_path = probe_dir / f"{var}.npy"
            np.save(output_path, data)
        
        # Save probe position
        np.save(probe_dir / "position.npy", probe_pos[p])
    
    print(f"✓ Saved data for {num_probes} probes to {probe_data_dir}")
    
    # Save probe positions as single file
    positions_file = output_dir / "probe_positions.npy"
    np.save(positions_file, probe_pos)
    print(f"✓ Saved all probe positions to {positions_file}")
    
    # Save simulation data
    print("\nSaving simulation data...")
    sim_dir = output_dir / "simulation"
    sim_dir.mkdir(exist_ok=True)
    
    for var, data in sim_data.items():
        output_path = sim_dir / f"{var}.npy"
        np.save(output_path, data)
        print(f"  ✓ {var} -> {output_path}")
    
    # Create visualization of probe positions
    print("\n" + "=" * 70)
    print("CREATING PROBE POSITION VISUALIZATION")
    print("=" * 70)
    
    visualize_probe_positions(probe_pos, nx, nz, output_dir, field_list, sim_data)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("PROBE DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nNumber of probes: {len(probe_list)}")
    print(f"Fields per probe: {field_list}")
    print(f"Additional variables: t_array")
    
    print("\nSample probe data dimensions (Probe 0):")
    for var, data in probe_data['0'].items():
        print(f"  {var:15s}: {data.shape}")
    
    print("\nSimulation data dimensions:")
    for var, data in sim_data.items():
        print(f"  {var:15s}: {data.shape}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)


def visualize_probe_positions(probe_pos, nx, nz, output_dir, field_list, sim_data):
    """
    Create visualization showing where probes are positioned in the simulation domain.
    """
    output_dir = Path(output_dir)
    
    # Extract positions
    x_positions = [probe_pos[p]['x'] for p in sorted(probe_pos.keys(), key=int)]
    z_positions = [probe_pos[p]['z'] for p in sorted(probe_pos.keys(), key=int)]
    probe_ids = [int(p) for p in sorted(probe_pos.keys(), key=int)]
    
    # Create figure with multiple subplots
    n_fields = len(field_list)
    fig, axes = plt.subplots(1, n_fields + 1, figsize=(6 * (n_fields + 1), 5))
    
    if n_fields == 1:
        axes = [axes]
    
    # Plot probe positions on grid
    ax = axes[0]
    ax.scatter(x_positions, z_positions, c='red', s=100, marker='x', linewidths=2, zorder=5)
    ax.set_xlim(0, int(nx))
    ax.set_ylim(0, int(nz))
    ax.set_xlabel('X position (grid index)')
    ax.set_ylabel('Z position (grid index)')
    ax.set_title(f'Probe Positions (n={len(probe_ids)})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Annotate some probes
    for i, (x, z, pid) in enumerate(zip(x_positions, z_positions, probe_ids)):
        if i % 8 == 0:  # Annotate every 8th probe to avoid clutter
            ax.annotate(f'P{pid}', (x, z), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
    
    # Plot probe positions overlaid on field data (first timestep)
    for idx, field in enumerate(field_list):
        ax = axes[idx + 1]
        
        # Get first timestep of simulation data
        # Assuming shape is (t, x, y, z) or similar
        if field in sim_data:
            data = sim_data[field]
            
            # Take first timestep and middle y-slice if 4D
            if len(data.shape) == 4:
                # (t, x, y, z) -> take t=0, y=middle
                field_slice = data[0, :, data.shape[2]//2, :]
            elif len(data.shape) == 3:
                # (t, x, z) -> take t=0
                field_slice = data[0, :, :]
            else:
                field_slice = data[0] if len(data.shape) > 1 else data
            
            # Plot field
            im = ax.imshow(field_slice.T, origin='lower', aspect='auto', cmap='viridis', 
                          extent=[0, int(nx), 0, int(nz)])
            plt.colorbar(im, ax=ax, label=field)
            
            # Overlay probe positions
            ax.scatter(x_positions, z_positions, c='red', s=50, marker='x', 
                      linewidths=2, zorder=5, alpha=0.8)
            
            ax.set_xlabel('X position')
            ax.set_ylabel('Z position')
            ax.set_title(f'{field} (t=0) with Probe Positions')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "probe_positions_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved probe position visualization to {output_path}")
    plt.close()
    
    # Create detailed position plot
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x_positions, z_positions, c=probe_ids, s=200, 
                        cmap='tab20', marker='o', edgecolors='black', linewidths=1.5)
    
    # Annotate all probes
    for x, z, pid in zip(x_positions, z_positions, probe_ids):
        ax.annotate(f'{pid}', (x, z), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
    
    ax.set_xlim(-5, int(nx) + 5)
    ax.set_ylim(-5, int(nz) + 5)
    ax.set_xlabel('X position (grid index)', fontsize=12)
    ax.set_ylabel('Z position (grid index)', fontsize=12)
    ax.set_title(f'TCV Probe Positions (n={len(probe_ids)}, nx={nx}, nz={nz})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.colorbar(scatter, ax=ax, label='Probe ID')
    
    output_path = output_dir / "probe_positions_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed probe position plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract probe data from TCV BOUT++ files"
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
        default="/dtu/blackhole/1b/223803/tcv_probe_data",
        help="Output directory for probe data"
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["n", "phi"],
        help="Field variables to extract"
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=64,
        help="Number of probes (0 to num_probes-1)"
    )
    
    args = parser.parse_args()
    
    extract_probe_data(
        bout_path=args.bout_path,
        output_dir=args.output,
        field_list=args.fields,
        num_probes=args.num_probes,
    )


if __name__ == "__main__":
    main()
