#!/usr/bin/env python3
"""
Neural Radial Basis Function Network for Probe-to-Field Reconstruction.

Combines classical RBF interpolation with deep learning refinement:
1. Learnable RBF centers and widths
2. RBF-based initial reconstruction
3. CNN refinement network for residual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class LearnableRBFLayer(nn.Module):
    """
    Learnable Radial Basis Function layer.
    
    Instead of fixed RBF interpolation, learns optimal centers and widths.
    
    Parameters
    ----------
    num_probes : int
        Number of input probe measurements
    num_rbf_centers : int
        Number of RBF centers to learn
    spatial_size : tuple
        Output spatial size (H, W)
    """
    
    def __init__(
        self,
        num_probes: int = 40,
        num_rbf_centers: int = 100,
        spatial_size: tuple = (256, 256),
    ):
        super().__init__()
        
        self.num_probes = num_probes
        self.num_rbf_centers = num_rbf_centers
        self.spatial_size = spatial_size
        
        # Learnable RBF centers (initialized randomly in spatial domain)
        H, W = spatial_size
        self.rbf_centers = nn.Parameter(
            torch.randn(num_rbf_centers, 2) * 0.5 + 0.5  # Normalized to [0, 1]
        )
        
        # Learnable RBF widths (one per center)
        self.rbf_widths = nn.Parameter(
            torch.ones(num_rbf_centers) * 0.1
        )
        
        # Learnable weights: map probe values to RBF activations
        self.probe_to_rbf = nn.Linear(num_probes, num_rbf_centers)
        
        # Create pixel coordinate grid (normalized to [0, 1])
        y_coords = torch.linspace(0, 1, H)
        x_coords = torch.linspace(0, 1, W)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixel_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        self.register_buffer('pixel_coords', pixel_coords)
    
    def forward(self, probe_values: torch.Tensor, probe_coords: torch.Tensor) -> torch.Tensor:
        """
        RBF interpolation from probes to dense field.
        
        Parameters
        ----------
        probe_values : torch.Tensor
            Probe measurements, shape (B, num_probes)
        probe_coords : torch.Tensor
            Probe coordinates (normalized [0, 1]), shape (num_probes, 2)
        
        Returns
        -------
        field : torch.Tensor
            Interpolated field, shape (B, H, W)
        """
        B = probe_values.shape[0]
        H, W = self.spatial_size
        
        # Map probe values to RBF weights
        rbf_weights = self.probe_to_rbf(probe_values)  # (B, num_rbf_centers)
        
        # Compute RBF activations for each pixel
        # Expand for broadcasting: pixel_coords (H, W, 2), rbf_centers (num_centers, 2)
        pixel_coords_expanded = self.pixel_coords.unsqueeze(2)  # (H, W, 1, 2)
        rbf_centers_expanded = self.rbf_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_centers, 2)
        
        # Compute squared distances
        distances_sq = torch.sum((pixel_coords_expanded - rbf_centers_expanded) ** 2, dim=-1)  # (H, W, num_centers)
        
        # Gaussian RBF: exp(-d^2 / (2 * width^2))
        widths_sq = self.rbf_widths ** 2
        rbf_activations = torch.exp(-distances_sq / (2 * widths_sq.unsqueeze(0).unsqueeze(0) + 1e-6))  # (H, W, num_centers)
        
        # Weight RBF activations with learned weights
        # rbf_activations: (H, W, num_centers), rbf_weights: (B, num_centers)
        field = torch.einsum('hwc,bc->bhw', rbf_activations, rbf_weights)  # (B, H, W)
        
        return field


class CNNRefinement(nn.Module):
    """
    CNN-based refinement network.
    
    Takes RBF interpolation and learns residual corrections.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.refine = nn.Sequential(
            # Initial processing
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Output: residual correction
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, rbf_output: torch.Tensor) -> torch.Tensor:
        """
        Refine RBF interpolation.
        
        Parameters
        ----------
        rbf_output : torch.Tensor
            Initial RBF interpolation, shape (B, C, H, W)
        
        Returns
        -------
        refined : torch.Tensor
            Refined field, shape (B, C, H, W)
        """
        residual = self.refine(rbf_output)
        refined = rbf_output + residual  # Residual learning
        
        return refined


class NeuralRBF(nn.Module):
    """
    Neural Radial Basis Function Network.
    
    Combines learnable RBF interpolation with CNN refinement.
    
    Parameters
    ----------
    num_probes : int
        Number of probes
    num_vars : int
        Number of variables
    num_rbf_centers : int
        Number of RBF centers to learn
    spatial_size : tuple
        Output spatial size
    base_channels : int
        Base channels for refinement CNN
    """
    
    def __init__(
        self,
        num_probes: int = 40,
        num_vars: int = 4,
        num_rbf_centers: int = 100,
        spatial_size: tuple = (256, 256),
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.num_probes = num_probes
        self.num_vars = num_vars
        self.spatial_size = spatial_size
        
        # One RBF layer per variable
        self.rbf_layers = nn.ModuleList([
            LearnableRBFLayer(num_probes, num_rbf_centers, spatial_size)
            for _ in range(num_vars)
        ])
        
        # CNN refinement
        self.refinement = CNNRefinement(num_vars, base_channels)
    
    def forward(
        self, 
        probe_values: torch.Tensor, 
        probe_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        probe_values : torch.Tensor
            Probe measurements, shape (B, num_vars, num_probes)
        probe_coords : torch.Tensor
            Probe coordinates (normalized), shape (num_probes, 2)
        
        Returns
        -------
        rbf_output : torch.Tensor
            RBF interpolation, shape (B, num_vars, H, W)
        refined_output : torch.Tensor
            Refined output, shape (B, num_vars, H, W)
        """
        B = probe_values.shape[0]
        
        # RBF interpolation for each variable
        rbf_fields = []
        for var_idx in range(self.num_vars):
            probe_vals_var = probe_values[:, var_idx, :]  # (B, num_probes)
            rbf_field = self.rbf_layers[var_idx](probe_vals_var, probe_coords)  # (B, H, W)
            rbf_fields.append(rbf_field)
        
        rbf_output = torch.stack(rbf_fields, dim=1)  # (B, num_vars, H, W)
        
        # CNN refinement
        refined_output = self.refinement(rbf_output)
        
        return rbf_output, refined_output


def neural_rbf_loss(
    rbf_output: torch.Tensor,
    refined_output: torch.Tensor,
    target: torch.Tensor,
    rbf_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss for Neural RBF.
    
    Penalizes both RBF and refined output.
    
    Parameters
    ----------
    rbf_output : torch.Tensor
        RBF interpolation
    refined_output : torch.Tensor
        Refined output
    target : torch.Tensor
        Ground truth
    rbf_weight : float
        Weight for RBF loss (encourages good initial interpolation)
    
    Returns
    -------
    total_loss : torch.Tensor
    rbf_loss : torch.Tensor
    refinement_loss : torch.Tensor
    """
    rbf_loss = F.mse_loss(rbf_output, target)
    refinement_loss = F.mse_loss(refined_output, target)
    
    total_loss = rbf_weight * rbf_loss + refinement_loss
    
    return total_loss, rbf_loss, refinement_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Neural RBF...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = NeuralRBF(
        num_probes=40,
        num_vars=4,
        num_rbf_centers=100,
        spatial_size=(256, 256),
        base_channels=32,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    B = 2
    probe_values = torch.randn(B, 4, 40).to(device)
    probe_coords = torch.rand(40, 2).to(device)  # Normalized coordinates
    
    rbf_output, refined_output = model(probe_values, probe_coords)
    
    print(f"\nInput probe values: {probe_values.shape}")
    print(f"Probe coordinates: {probe_coords.shape}")
    print(f"RBF output: {rbf_output.shape}")
    print(f"Refined output: {refined_output.shape}")
    
    # Test loss
    target = torch.randn_like(refined_output)
    loss, rbf_loss, ref_loss = neural_rbf_loss(rbf_output, refined_output, target)
    
    print(f"\nLosses:")
    print(f"  Total: {loss.item():.4f}")
    print(f"  RBF: {rbf_loss.item():.4f}")
    print(f"  Refinement: {ref_loss.item():.4f}")
    
    # Check RBF centers
    print(f"\nLearned RBF centers range:")
    for i, rbf_layer in enumerate(model.rbf_layers):
        centers = rbf_layer.rbf_centers
        print(f"  Var {i}: x=[{centers[:, 0].min():.3f}, {centers[:, 0].max():.3f}], "
              f"z=[{centers[:, 1].min():.3f}, {centers[:, 1].max():.3f}]")
        print(f"  Widths: [{rbf_layer.rbf_widths.min():.3f}, {rbf_layer.rbf_widths.max():.3f}]")
    
    print("\n✓ Neural RBF test passed!")
