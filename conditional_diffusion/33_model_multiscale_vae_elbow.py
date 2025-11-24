#!/usr/bin/env python3
"""
Multi-Scale VAE with Elbow Loss (Robust Reconstruction).

Features:
- Spatial VAE (no temporal component in base model)
- Multi-scale outputs (64, 128, 256)
- Elbow Loss (SmoothL1Loss) for robust reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class Encoder(nn.Module):
    """
    Encoder with multi-scale feature extraction.
    """
    def __init__(
        self,
        in_channels: int = 2,  # n, phi
        latent_dim: int = 256,
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Progressive downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Latent distribution
        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        
        h = self.global_pool(h).flatten(1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class ResidualRefinementBlock(nn.Module):
    """Refinement block that learns residual correction."""
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        if channels != out_channels:
            self.channel_adjust = nn.Conv2d(channels, out_channels, 1)
        else:
            self.channel_adjust = nn.Identity()
    
    def forward(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        # Bilinear upsample
        x_upsampled = F.interpolate(x, size=(target_size, target_size), 
                                     mode='bilinear', align_corners=False)
        
        x_base = self.channel_adjust(x_upsampled)
        residual = self.refine(x_upsampled)
        
        return x_base + residual


class ProgressiveDecoder(nn.Module):
    """Progressive decoder with multi-scale outputs."""
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 2,
        base_channels: int = 32,
        target_sizes: List[int] = [64, 128, 256],
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.target_sizes = target_sizes
        
        # Initial projection
        init_size = target_sizes[0] // 4
        self.fc = nn.Linear(latent_dim, base_channels * 8 * init_size * init_size)
        self.init_size = init_size
        self.base_channels = base_channels
        
        self.refinement_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_channels = base_channels * 8
        for i, target_size in enumerate(target_sizes):
            # Determine output channels for this block
            # We want to reduce channels progressively: 256 -> 128 -> 64 -> 32
            next_channels = max(base_channels, current_channels // 2)
            
            self.refinement_blocks.append(
                ResidualRefinementBlock(current_channels, next_channels)
            )
            
            self.to_rgb.append(
                nn.Conv2d(next_channels, out_channels, 1)
            )
            
            current_channels = next_channels
            
    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, self.init_size, self.init_size)
        
        outputs = {}
        
        for i, target_size in enumerate(self.target_sizes):
            h = self.refinement_blocks[i](h, target_size)
            outputs[target_size] = self.to_rgb[i](h)
            
        return outputs


class MultiScaleVAE(nn.Module):
    """
    Multi-Scale VAE with Elbow Loss.
    """
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 256,
        base_channels: int = 32,
        target_sizes: List[int] = [64, 128, 256],
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = ProgressiveDecoder(latent_dim, in_channels, base_channels, target_sizes)
        self.latent_dim = latent_dim
        self.target_sizes = target_sizes
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decoder(z)
        return outputs, mu, logvar


def elbow_loss(
    outputs: Dict[int, torch.Tensor],
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-4,
    scale_weights: Dict[int, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Elbow Loss (Robust Reconstruction) + KL Divergence.
    
    Uses SmoothL1Loss (Huber Loss) which is robust to outliers ("elbow" shape).
    """
    if scale_weights is None:
        scale_weights = {size: 1.0 for size in outputs.keys()}
    
    scale_losses = {}
    total_recon_loss = 0.0
    total_weight = sum(scale_weights.values())
    
    for size, output in outputs.items():
        # Downsample target to match this scale
        if target.shape[-1] != size:
            target_scaled = F.interpolate(
                target, size=(size, size), mode='bilinear', align_corners=False
            )
        else:
            target_scaled = target
            
        # Elbow Loss (Smooth L1)
        # beta=1.0 means L2 for |x|<1, L1 for |x|>1
        scale_loss = F.smooth_l1_loss(output, target_scaled, reduction='mean', beta=1.0)
        
        scale_losses[size] = scale_loss
        total_recon_loss += scale_weights[size] * scale_loss
    
    total_recon_loss = total_recon_loss / total_weight
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.size(0)
    
    loss = total_recon_loss + kl_weight * kl_loss
    
    return loss, total_recon_loss, kl_loss, scale_losses


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
