#!/usr/bin/env python3
"""
Visualize extracted BOUT++ data and probe sampling results.

Creates plots to verify data extraction and probe placement.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_field_with_probes(
    field_data: np.ndarray,
    probe_coords: np.ndarray,
    title: str = "Field with Probe Locations",
    timestep: int = -1,
    cmap: str = "plasma",
    save_path: Path = None,
):
    """
    Plot a 2D field with probe locations overlaid.
    
    Parameters
    ----------
    field_data : ndarray of shape (time, x, z) or (x, z)
        Field data to visualize
    probe_coords : ndarray of shape (num_probes, 2)
        Probe coordinates as (x, z) pairs
    title : str
        Plot title
    timestep : int
        Which timestep to plot (for 3D data)
    cmap : str
        Colormap name
    save_path : Path, optional
        If provided, save figure to this path
    """
    # Extract frame
    if field_data.ndim == 3:
        frame = field_data[timestep, :, :]
    else:
        frame = field_data
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot field
    im = ax.imshow(
        frame.T,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title, rotation=270, labelpad=20)
    
    # Overlay probe locations
    probe_x = probe_coords[:, 0]
    probe_z = probe_coords[:, 1]
    ax.scatter(
        probe_x, probe_z,
        c='red',
        s=50,
        marker='x',
        linewidths=2,
        label=f'{len(probe_coords)} probes',
        zorder=10
    )
    
    ax.set_xlabel('x (radial direction)')
    ax.set_ylabel('z (along field line)')
    ax.set_title(f'{title} (timestep {timestep})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_probe_timeseries(
    probe_values: dict[str, np.ndarray],
    probe_index: int = 0,
    save_path: Path = None,
):
    """
    Plot time series of probe measurements for all variables.
    
    Parameters
    ----------
    probe_values : dict
        Dictionary mapping variable names to probe value arrays (time, num_probes)
    probe_index : int
        Which probe to plot
    save_path : Path, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(len(probe_values), 1, figsize=(10, 3*len(probe_values)))
    
    if len(probe_values) == 1:
        axes = [axes]
    
    for ax, (var_name, values) in zip(axes, probe_values.items()):
        time_steps = np.arange(values.shape[0])
        ax.plot(time_steps, values[:, probe_index], linewidth=1.5)
        ax.set_xlabel('Time step')
        ax.set_ylabel(var_name)
        ax.set_title(f'{var_name} at probe {probe_index}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_probe_spatial_profile(
    probe_values: dict[str, np.ndarray],
    probe_coords: np.ndarray,
    timestep: int = -1,
    save_path: Path = None,
):
    """
    Plot spatial profile of probe measurements at a given timestep.
    
    Parameters
    ----------
    probe_values : dict
        Dictionary mapping variable names to probe value arrays (time, num_probes)
    probe_coords : ndarray of shape (num_probes, 2)
        Probe coordinates
    timestep : int
        Which timestep to plot
    save_path : Path, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(len(probe_values), 1, figsize=(10, 3*len(probe_values)))
    
    if len(probe_values) == 1:
        axes = [axes]
    
    probe_x = probe_coords[:, 0]
    
    for ax, (var_name, values) in zip(axes, probe_values.items()):
        frame_values = values[timestep, :]
        ax.plot(probe_x, frame_values, 'o-', linewidth=1.5, markersize=5)
        ax.set_xlabel('x position (radial)')
        ax.set_ylabel(var_name)
        ax.set_title(f'{var_name} spatial profile at timestep {timestep}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def visualize_results(
    data_dir: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    variables: list[str] = None,
    timestep: int = -1,
    probe_index: int = 0,
):
    """
    Create visualization plots for extracted data and probes.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing extracted field .npy files
    probe_dir : str or Path
        Directory containing probe data
    output_dir : str or Path
        Directory to save plots
    variables : list of str, optional
        Variables to visualize (default: ['n', 'te', 'ti', 'phi'])
    timestep : int
        Which timestep to visualize
    probe_index : int
        Which probe to plot for time series
    """
    if variables is None:
        variables = ['n', 'te', 'ti', 'phi']
    
    data_dir = Path(data_dir)
    probe_dir = Path(probe_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations...")
    print(f"Data dir: {data_dir}")
    print(f"Probe dir: {probe_dir}")
    print(f"Output dir: {output_dir}")
    
    # Load probe coordinates
    probe_coords_path = probe_dir / "probe_coordinates.npy"
    if not probe_coords_path.exists():
        print(f"ERROR: Probe coordinates not found at {probe_coords_path}")
        return
    
    probe_coords = np.load(probe_coords_path)
    print(f"\nProbe coordinates shape: {probe_coords.shape}")
    
    # Load probe values
    probe_values = {}
    for var_name in variables:
        probe_path = probe_dir / f"probe_{var_name}.npy"
        if probe_path.exists():
            probe_values[var_name] = np.load(probe_path)
            print(f"Loaded probe_{var_name}: {probe_values[var_name].shape}")
        else:
            print(f"WARNING: {probe_path} not found, skipping")
    
    # Plot 1: Fields with probe locations
    print("\n=== Creating field plots with probe overlays ===")
    for var_name in variables:
        field_path = data_dir / f"{var_name}.npy"
        if not field_path.exists():
            print(f"WARNING: {field_path} not found, skipping")
            continue
        
        field_data = np.load(field_path)
        print(f"\nPlotting {var_name} field...")
        
        plot_field_with_probes(
            field_data,
            probe_coords,
            title=var_name,
            timestep=timestep,
            save_path=output_dir / f"field_{var_name}_with_probes.png"
        )
    
    # Plot 2: Probe time series
    if probe_values:
        print("\n=== Creating probe time series plots ===")
        plot_probe_timeseries(
            probe_values,
            probe_index=probe_index,
            save_path=output_dir / f"probe_timeseries_idx{probe_index}.png"
        )
    
    # Plot 3: Probe spatial profiles
    if probe_values:
        print("\n=== Creating spatial profile plots ===")
        plot_probe_spatial_profile(
            probe_values,
            probe_coords,
            timestep=timestep,
            save_path=output_dir / f"probe_spatial_profile_t{timestep}.png"
        )
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BOUT++ data and probe sampling"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/bout_data",
        help="Directory containing extracted field .npy files"
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default="/dtu/blackhole/1b/223803/probe_data",
        help="Directory containing probe data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dtu/blackhole/1b/223803/visualizations",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["n", "te", "ti", "phi"],
        help="Variables to visualize"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=-1,
        help="Which timestep to visualize (default: -1 = last)"
    )
    parser.add_argument(
        "--probe-index",
        type=int,
        default=0,
        help="Which probe to plot for time series"
    )
    
    args = parser.parse_args()
    
    visualize_results(
        data_dir=args.data_dir,
        probe_dir=args.probe_dir,
        output_dir=args.output,
        variables=args.variables,
        timestep=args.timestep,
        probe_index=args.probe_index,
    )


if __name__ == "__main__":
    main()
