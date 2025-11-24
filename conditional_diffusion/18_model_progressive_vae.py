#!/usr/bin/env python3
"""
Progressive Multi-Scale VAE with Residual Refinement.

Implements:
1. Progressive upsampling: 64 → 128 → 256 → 512 (or 1024)
2. Residual learning: Network predicts correction to bilinear interpolation
3. Multi-scale loss: Penalizes errors at each resolution level
4. TEMPORAL: ConvLSTM encoder for processing probe sequences over time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Optional


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for temporal processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
    
    def forward(self, input_tensor, h_cur, c_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using ConvLSTM to process probe sequences.
    
    Takes a sequence of probe measurements over time and encodes them
    into a latent representation.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (variables)
    latent_dim : int
        Latent dimension
    base_channels : int
        Base channel count
    hidden_dims : list of int
        Hidden dimensions for ConvLSTM layers
    sequence_length : int
        Number of timesteps to process
    """
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        base_channels: int = 32,
        hidden_dims: List[int] = [64, 128],
        sequence_length: int = 10,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.hidden_dims = hidden_dims
        
        # Initial spatial processing
        self.spatial_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM layers
        self.conv_lstm_cells = nn.ModuleList()
        input_dim = base_channels * 2
        for h_dim in hidden_dims:
            self.conv_lstm_cells.append(ConvLSTMCell(input_dim, h_dim, kernel_size=3))
            input_dim = h_dim
        
        # Global pooling + latent projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal sequence.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (B, T, C, H, W)
        
        Returns
        -------
        mu : torch.Tensor
            Latent mean
        logvar : torch.Tensor
            Latent log variance
        """
        B, T, C, H, W = x.shape
        
        # Initialize hidden states
        hidden_states = [None] * len(self.hidden_dims)
        
        # Process sequence
        for t in range(T):
            # Spatial embedding
            x_t = x[:, t]  # (B, C, H, W)
            feat = self.spatial_embed(x_t)  # (B, base_channels*2, H, W)
            
            # ConvLSTM processing
            input_t = feat
            for i, cell in enumerate(self.conv_lstm_cells):
                h, c = hidden_states[i] if hidden_states[i] is not None else (None, None)
                
                if h is None:
                    # Initialize hidden state
                    h = torch.zeros(B, self.hidden_dims[i], H, W).to(x.device)
                    c = torch.zeros(B, self.hidden_dims[i], H, W).to(x.device)
                
                h_next, c_next = cell(input_t, h, c)
                hidden_states[i] = (h_next, c_next)
                input_t = h_next
        
        # Use final hidden state from last layer
        final_h = hidden_states[-1][0]  # (B, hidden_dims[-1], H, W)
        
        # Pool and project to latent space
        pooled = self.global_pool(final_h).flatten(1)  # (B, hidden_dims[-1])
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        return mu, logvar


class Encoder(nn.Module):
    """
    Encoder with multi-scale feature extraction.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    latent_dim : int
        Latent dimension
    base_channels : int
        Base channel count
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
        
        # Progressive downsampling
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
    """
    Refinement block that learns residual correction to bilinear interpolation.
    
    Takes coarse input, upsamples it with bilinear, then adds learned residual.
    """
    
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
        
        # Separate path for bilinear upsampled features
        if channels != out_channels:
            self.channel_adjust = nn.Conv2d(channels, out_channels, 1)
        else:
            self.channel_adjust = nn.Identity()
    
    def forward(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, C, H, W)
        target_size : int
            Target spatial size
        
        Returns
        -------
        refined : torch.Tensor
            Refined features at target size
        """
        # Bilinear upsample
        x_upsampled = F.interpolate(x, size=(target_size, target_size), 
                                     mode='bilinear', align_corners=False)
        
        # Adjust channels if needed
        x_base = self.channel_adjust(x_upsampled)
        
        # Learn residual correction
        residual = self.refine(x_upsampled)
        
        # Add residual to base
        refined = x_base + residual
        
        return refined


class ProgressiveDecoder(nn.Module):
    """
    Progressive decoder with multi-scale outputs.
    
    Generates intermediate outputs at 64, 128, 256, ... resolutions.
    Each stage learns residual corrections to bilinear upsampling.
    
    Parameters
    ----------
    latent_dim : int
        Latent dimension
    out_channels : int
        Number of output channels
    base_channels : int
        Base channel count
    target_sizes : list of int
        Progressive output sizes (e.g., [64, 128, 256, 512])
    intermediate_noise_scale : float
        Noise injection scale at intermediate stages
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 4,
        base_channels: int = 32,
        target_sizes: List[int] = [64, 128, 256, 512],
        intermediate_noise_scale: float = 0.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.target_sizes = target_sizes
        self.intermediate_noise_scale = intermediate_noise_scale
        
        # Initial projection: latent → smallest spatial size
        init_size = target_sizes[0] // 4  # Start even smaller
        self.fc = nn.Linear(latent_dim, base_channels * 8 * init_size * init_size)
        self.init_size = init_size
        self.base_channels = base_channels
        
        # Progressive refinement blocks
        self.refinement_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()  # Output heads at each scale
        
        current_channels = base_channels * 8
        for i, target_size in enumerate(target_sizes):
            # Channel reduction as we go up in resolution
            next_channels = max(base_channels, current_channels // 2)
            
            # Refinement block
            refine_block = ResidualRefinementBlock(current_channels, next_channels)
            self.refinement_blocks.append(refine_block)
            
            # RGB/output head for this resolution
            to_rgb_layer = nn.Conv2d(next_channels, out_channels, 1)
            self.to_rgb.append(to_rgb_layer)
            
            current_channels = next_channels
    
    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Progressive decoding with multi-scale outputs.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent code, shape (B, latent_dim)
        
        Returns
        -------
        outputs : dict
            Dictionary mapping resolution → output tensor
            Keys are resolution sizes (64, 128, 256, etc.)
        """
        # Project to initial spatial features
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, self.init_size, self.init_size)
        
        outputs = {}
        
        # Progressive refinement
        for i, target_size in enumerate(self.target_sizes):
            # Refine to next resolution
            h = self.refinement_blocks[i](h, target_size)
            
            # Optional noise injection (training only)
            if self.training and self.intermediate_noise_scale > 0:
                h = h + torch.randn_like(h) * self.intermediate_noise_scale
            
            # Generate output at this resolution
            output = self.to_rgb[i](h)
            outputs[target_size] = output
        
        return outputs


class ProgressiveVAE(nn.Module):
    """
    Progressive Multi-Scale VAE.
    
    Features:
    - Progressive upsampling at multiple scales
    - Residual learning on bilinear interpolation
    - Multi-scale outputs for multi-scale loss
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    latent_dim : int
        Latent dimension
    base_channels : int
        Base channel count
    target_sizes : list of int
        Progressive output sizes
    intermediate_noise_scale : float
        Noise injection scale
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        base_channels: int = 32,
        target_sizes: List[int] = [64, 128, 256, 512],
        intermediate_noise_scale: float = 0.0,
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = ProgressiveDecoder(
            latent_dim, in_channels, base_channels, 
            target_sizes, intermediate_noise_scale
        )
        self.latent_dim = latent_dim
        self.target_sizes = target_sizes
        self.intermediate_noise_scale = intermediate_noise_scale
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (B, C, H, W)
        
        Returns
        -------
        outputs : dict
            Multi-scale reconstructions
        mu : torch.Tensor
            Latent mean
        logvar : torch.Tensor
            Latent log variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decoder(z)
        
        return outputs, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return self.decoder(z)


def progressive_vae_loss(
    outputs: Dict[int, torch.Tensor],
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-4,
    scale_weights: Dict[int, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Multi-scale VAE loss.
    
    Computes reconstruction loss at each resolution and combines them.
    
    Parameters
    ----------
    outputs : dict
        Multi-scale outputs from decoder
    target : torch.Tensor
        Target image at full resolution
    mu : torch.Tensor
        Latent mean
    logvar : torch.Tensor
        Latent log variance
    kl_weight : float
        KL divergence weight
    scale_weights : dict, optional
        Weight for each scale (default: equal weights)
    
    Returns
    -------
    loss : torch.Tensor
        Total loss
    recon_loss : torch.Tensor
        Total reconstruction loss
    kl_loss : torch.Tensor
        KL divergence
    scale_losses : dict
        Individual reconstruction loss at each scale
    """
    # Default equal weights for all scales
    if scale_weights is None:
        scale_weights = {size: 1.0 for size in outputs.keys()}
    
    # Compute reconstruction loss at each scale
    scale_losses = {}
    total_recon_loss = 0.0
    total_weight = sum(scale_weights.values())
    
    for size, output in outputs.items():
        # Downsample target to match this scale
        target_scaled = F.interpolate(
            target, size=(size, size), 
            mode='bilinear', align_corners=False
        )
        
        # MSE loss at this scale
        loss_scale = F.mse_loss(output, target_scaled)
        scale_losses[size] = loss_scale
        
        # Weighted contribution
        weight = scale_weights.get(size, 1.0)
        total_recon_loss += weight * loss_scale
    
    # Normalize by total weight
    total_recon_loss = total_recon_loss / total_weight
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.size(0)
    
    # Total loss
    loss = total_recon_loss + kl_weight * kl_loss
    
    return loss, total_recon_loss, kl_loss, scale_losses


class TemporalProgressiveVAE(nn.Module):
    """
    Temporal Progressive VAE.
    
    Processes a sequence of probe measurements over time using ConvLSTM,
    then generates multi-scale reconstruction of the current timestep.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (variables)
    latent_dim : int
        Latent dimension
    base_channels : int
        Base channel count
    target_sizes : list of int
        Progressive output sizes
    hidden_dims : list of int
        Hidden dimensions for ConvLSTM
    sequence_length : int
        Number of timesteps to process
    intermediate_noise_scale : float
        Noise injection scale
    """
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        base_channels: int = 32,
        target_sizes: List[int] = [64, 128, 256],
        hidden_dims: List[int] = [64, 128],
        sequence_length: int = 10,
        intermediate_noise_scale: float = 0.0,
    ):
        super().__init__()
        
        self.temporal_encoder = TemporalEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            hidden_dims=hidden_dims,
            sequence_length=sequence_length,
        )
        
        self.decoder = ProgressiveDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            target_sizes=target_sizes,
            intermediate_noise_scale=intermediate_noise_scale,
        )
        
        self.latent_dim = latent_dim
        self.target_sizes = target_sizes
        self.sequence_length = sequence_length
        self.intermediate_noise_scale = intermediate_noise_scale
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (B, T, C, H, W)
        
        Returns
        -------
        outputs : dict
            Multi-scale reconstructions of current timestep
        mu : torch.Tensor
            Latent mean
        logvar : torch.Tensor
            Latent log variance
        """
        mu, logvar = self.temporal_encoder(x)
        z = self.reparameterize(mu, logvar)
        outputs = self.decoder(z)
        
        return outputs, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.temporal_encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return self.decoder(z)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Progressive VAE...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with different target sizes
    model = ProgressiveVAE(
        in_channels=4,
        latent_dim=256,
        base_channels=32,
        target_sizes=[64, 128, 256],
        intermediate_noise_scale=0.05,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Progressive scales: {model.target_sizes}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 256, 256).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    outputs, mu, logvar = model(x)
    
    print("\nMulti-scale outputs:")
    for size, output in outputs.items():
        print(f"  {size}×{size}: {output.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss, scale_losses = progressive_vae_loss(
        outputs, x, mu, logvar
    )
    
    print(f"\nLosses:")
    print(f"  Total: {loss.item():.4f}")
    print(f"  Reconstruction: {recon_loss.item():.4f}")
    print(f"  KL: {kl_loss.item():.4f}")
    print(f"  Per-scale:")
    for size, scale_loss in scale_losses.items():
        print(f"    {size}x{size}: {scale_loss.item():.4f}")
    
    print("\n✓ Model test passed!")
    
    # Test Temporal Progressive VAE
    print("\n" + "="*70)
    print("Testing Temporal Progressive VAE...")
    print("="*70)
    
    temporal_model = TemporalProgressiveVAE(
        in_channels=4,
        latent_dim=256,
        base_channels=32,
        target_sizes=[64, 128, 256],
        hidden_dims=[64, 128],
        sequence_length=10,
        intermediate_noise_scale=0.0,
    ).to(device)
    
    print(f"Temporal model parameters: {count_parameters(temporal_model):,}")
    print(f"Sequence length: {temporal_model.sequence_length}")
    print(f"Progressive scales: {temporal_model.target_sizes}")
    
    # Test forward pass with sequence
    batch_size = 2
    seq_len = 10
    x_seq = torch.randn(batch_size, seq_len, 4, 256, 256).to(device)
    
    print(f"\nInput sequence shape: {x_seq.shape}")
    
    outputs_temp, mu_temp, logvar_temp = temporal_model(x_seq)
    
    print("\nTemporal multi-scale outputs:")
    for size, output in outputs_temp.items():
        print(f"  {size}x{size}: {output.shape}")
    
    print(f"\nLatent: mu={mu_temp.shape}, logvar={logvar_temp.shape}")
    
    # Test loss with target (last frame)
    target_seq = x_seq[:, -1]  # Use last frame as target
    loss_temp, recon_temp, kl_temp, scale_losses_temp = progressive_vae_loss(
        outputs_temp, target_seq, mu_temp, logvar_temp
    )
    
    print(f"\nTemporal losses:")
    print(f"  Total: {loss_temp.item():.4f}")
    print(f"  Reconstruction: {recon_temp.item():.4f}")
    print(f"  KL: {kl_temp.item():.4f}")
    
    print("\n✓ Temporal Progressive VAE test passed!")
