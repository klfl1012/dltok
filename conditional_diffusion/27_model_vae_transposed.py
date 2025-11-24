#!/usr/bin/env python3
"""
VAE with Transposed Convolutions for upsampling.

This variant uses:
- Transposed convolutions (ConvTranspose2d) instead of bilinear upsampling
- More aggressive data augmentation with multiple probe samples per image
- Designed for heavy augmentation training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Convolutional encoder network.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (number of variables)
    latent_dim : int
        Dimensionality of latent space
    base_channels : int
        Number of channels in first conv layer
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Convolutional layers with downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /16
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution."""
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        
        # Global pooling
        h = self.global_pool(h).flatten(1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class DecoderTransposed(nn.Module):
    """
    Decoder with transposed convolutions for upsampling.
    
    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent space
    out_channels : int
        Number of output channels
    base_channels : int
        Base number of channels
    input_size : int
        Target output size
    intermediate_noise_scale : float
        Noise injection scale during training
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 4,
        base_channels: int = 32,
        input_size: int = 1024,
        intermediate_noise_scale: float = 0.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.intermediate_noise_scale = intermediate_noise_scale
        
        # Calculate spatial size at bottleneck
        self.spatial_size = input_size // 16
        
        # Project latent to spatial feature map
        self.fc = nn.Linear(latent_dim, base_channels * 8 * self.spatial_size * self.spatial_size)
        self.base_channels = base_channels
        
        # Transposed convolutions for upsampling (x2 each)
        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to image."""
        # Project and reshape
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, self.spatial_size, self.spatial_size)
        
        # Upsample with transposed convolutions and optional noise injection
        h = self.up1(h)
        if self.training and self.intermediate_noise_scale > 0:
            h = h + torch.randn_like(h) * self.intermediate_noise_scale
        
        h = self.up2(h)
        if self.training and self.intermediate_noise_scale > 0:
            h = h + torch.randn_like(h) * self.intermediate_noise_scale
        
        h = self.up3(h)
        if self.training and self.intermediate_noise_scale > 0:
            h = h + torch.randn_like(h) * self.intermediate_noise_scale
        
        h = self.up4(h)
        if self.training and self.intermediate_noise_scale > 0:
            h = h + torch.randn_like(h) * self.intermediate_noise_scale
        
        # Output
        x_recon = self.out_conv(h)
        
        return x_recon


class VAETransposed(nn.Module):
    """
    VAE with transposed convolutions.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    latent_dim : int
        Dimensionality of latent space
    base_channels : int
        Base number of channels
    input_size : int
        Input image spatial size
    intermediate_noise_scale : float
        Noise injection scale
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        base_channels: int = 32,
        input_size: int = 1024,
        intermediate_noise_scale: float = 0.0,
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = DecoderTransposed(latent_dim, in_channels, base_channels, input_size, intermediate_noise_scale)
        self.latent_dim = latent_dim
        self.intermediate_noise_scale = intermediate_noise_scale
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent code."""
        return self.decoder(z)


def vae_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE ELBO loss."""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x_target, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x_target.size(0)
    
    # Total loss
    loss = recon_loss + kl_weight * kl_loss
    
    return loss, recon_loss, kl_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing VAE with Transposed Convolutions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VAETransposed(
        in_channels=4,
        latent_dim=256,
        base_channels=32,
        input_size=1024,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 1024, 1024).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    x_recon, mu, logvar = model(x)
    
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    print("\n✓ Model test passed!")
