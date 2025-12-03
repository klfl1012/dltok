import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from abc import abstractmethod
from neuralop import FNO, TFNO
from losses import *
import matplotlib.pyplot as plt
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer # pip install denoising_diffusion_pytorch
from denoising_diffusion_pytorch.version import __version__
from tqdm.auto import tqdm

def divisible_by(numer, denom):
    return (numer % denom) == 0

class BaseModel(L.LightningModule):
    """Base class for all models with shared training/validation/test logic."""
    
    def __init__(self, learning_rate: float, loss_function: str, num_predictions_to_log: int = 1, log_images_every_n_epochs: int = 1, max_image_logging_epochs: int | None = None, enable_val_image_logging: bool = False, enable_inference_image_logging: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_name = loss_function
        self.num_predictions_to_log = num_predictions_to_log  # Number of sequence predictions to visualize
        self.log_images_every_n_epochs = max(1, int(log_images_every_n_epochs))
        self.max_image_logging_epochs = max_image_logging_epochs  # None = no global cap
        self._image_logging_epochs = 0
        self.enable_val_image_logging = enable_val_image_logging
        self.enable_inference_image_logging = enable_inference_image_logging
        
        self.validation_step_outputs = []
        
        # Map string names to concrete loss implementations.
        if loss_function in ("MSE", "L2", "LpLoss"):
            self.loss_function = Neuralop_LpLoss()
        elif loss_function == "MSE+Grad":
            self.loss_function = MSEWithGradientLoss()
        elif loss_function == "H1":
            self.loss_function = Neuralop_H1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        train_loss = self.loss_function(y_hat, y)
        
        log_dict = {
            f"train_{self.loss_name}_loss": train_loss,
            "learning_rate": self.learning_rate
        }
        self.log_dict(
            log_dict, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True,
            logger=True  
        )
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)
        
        self.log(
            f"val_{self.loss_name}_loss", 
            val_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        
        # Store outputs from first batch for end-of-epoch image logging
        if self.enable_val_image_logging and batch_idx == 0:
            # Store up to num_predictions_to_log samples
            num_samples = min(self.num_predictions_to_log, x.shape[0])
            for i in range(num_samples):
                self.validation_step_outputs.append({
                    'x': x[i:i+1].detach().cpu(),
                    'y_true': y[i:i+1].detach().cpu(),
                    'y_pred': y_hat[i:i+1].detach().cpu()
                })
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        
        # Log metrics
        self.log(
            f"test_{self.loss_name}_loss", 
            test_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        
        return test_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)

        # Optional image logging during inference
        if self.enable_inference_image_logging and batch_idx == 0 and self.logger is not None:
            # Log up to num_predictions_to_log samples from the first batch
            num_samples = min(self.num_predictions_to_log, x.shape[0])
            for i in range(num_samples):
                self._log_predictions_as_images(
                    x[i:i+1].detach().cpu(),
                    y[i:i+1].detach().cpu(),
                    y_hat[i:i+1].detach().cpu(),
                    prefix='inference',
                    sample_idx=i,
                )

        return y_hat, y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), logger=True)
    
    def on_validation_epoch_end(self):
        """Log images only on selected epochs to avoid flooding logs."""
        if not self.enable_val_image_logging:
            self.validation_step_outputs.clear()
            return

        should_log = False

        if self.max_image_logging_epochs is not None:
            if self._image_logging_epochs < self.max_image_logging_epochs:
                should_log = True
        else:
            should_log = True

        if should_log:
            if (int(self.current_epoch) % self.log_images_every_n_epochs) != 0:
                should_log = False

        if should_log and len(self.validation_step_outputs) > 0 and self.logger is not None:
            for idx, outputs in enumerate(self.validation_step_outputs):
                x = outputs['x']
                y_true = outputs['y_true']
                y_pred = outputs['y_pred']

                self._log_predictions_as_images(x, y_true, y_pred, prefix='val', sample_idx=idx)

            self._image_logging_epochs += 1
        
        # Clear stored outputs for next epoch
        self.validation_step_outputs.clear()
    
    def _log_predictions_as_images(self, x, y_true, y_pred, prefix='val', sample_idx=0):
        """
        Log predictions as images to TensorBoard or WandB.
        Each prediction consists of multiple timestep images (the full sequence length).
        
        Args:
            x: Input tensor [B, T, X, Y] where B=1 (single sample)
            y_true: Ground truth [B, T, X, Y]
            y_pred: Predictions [B, T, X, Y]
            prefix: Prefix for logging ('val' or 'test')
            sample_idx: Index of the sample being logged (for multiple predictions per epoch)
        """
        try:
            batch_idx = 0  # Always 0 since we store individual samples
            seq_len = y_true.shape[1]
            
            sample_true = y_true[batch_idx]
            sample_pred = y_pred[batch_idx]

            if sample_true.is_cuda:
                sample_true = sample_true.detach().cpu()
                sample_pred = sample_pred.detach().cpu()
            else:
                sample_true = sample_true.detach()
                sample_pred = sample_pred.detach()

            if sample_true.ndim == 3:
                y_true_np = sample_true.numpy()
                y_pred_np = sample_pred.numpy()
            elif sample_true.ndim == 4:
                # Multi-channel input; visualize the first channel by default.
                y_true_np = sample_true[:, 0].numpy()
                y_pred_np = sample_pred[:, 0].numpy()
            else:
                raise ValueError(
                    f'Unsupported tensor rank {sample_true.ndim} for image logging; expected 3 or 4 dimensions.'
                )
            
            timesteps_to_show = list(range(seq_len))
            num_timesteps = seq_len
            
            fig, axes = plt.subplots(3, num_timesteps, figsize=(4 * num_timesteps, 12))
            
            if num_timesteps == 1:
                axes = axes.reshape(-1, 1)
            
            for col_idx, t in enumerate(timesteps_to_show):
                true_frame = y_true_np[t]
                pred_frame = y_pred_np[t]
                error_frame = np.abs(true_frame - pred_frame)
                
                im0 = axes[0, col_idx].imshow(true_frame, cmap='viridis', aspect='auto')
                axes[0, col_idx].set_title(f'Ground Truth (t={t})')
                axes[0, col_idx].axis('off')
                plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046)
                
                im1 = axes[1, col_idx].imshow(pred_frame, cmap='viridis', aspect='auto')
                axes[1, col_idx].set_title(f'Prediction (t={t})')
                axes[1, col_idx].axis('off')
                plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046)
                
                im2 = axes[2, col_idx].imshow(error_frame, cmap='hot', aspect='auto')
                axes[2, col_idx].set_title(f'Abs Error (t={t})')
                axes[2, col_idx].axis('off')
                plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046)
            
            plt.tight_layout()
            
            # Use sample_idx in key name for multiple predictions
            key_suffix = f'_sample_{sample_idx}' if self.num_predictions_to_log > 1 else ''
            
            if isinstance(self.logger, TensorBoardLogger) and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(
                    f'{prefix}/predictions_sequence{key_suffix}',
                    fig,
                    global_step=self.current_epoch
                )
            elif isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    key=f'{prefix}/predictions_sequence{key_suffix}',
                    images=[fig],
                    step=self.current_epoch
                )
            
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log images: {e}")


class FNOModel(BaseModel):
    
    def __init__(
        self,
        n_modes: tuple = (16, 16, 16),
        hidden_channels: int = 64,
        in_channels: int = 4,
        out_channels: int = 1,
        n_layers: int = 4,
        non_linearity = F.gelu,
        stabilizer: str | None = None,
        norm: str | None = None,
        preactivation: bool = False,
        fno_skip: str = 'linear',
        separable: bool = False,
        factorization: str | None = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = 'factorized',
        decomposition_kwargs: dict | None = None,
        domain_padding: float | None = None,
        learning_rate: float = 1e-3,
        loss_function: str = "MSE",
        num_predictions_to_log: int = 1,
        log_images_every_n_epochs: int = 1,
        max_image_logging_epochs: int | None = None,
        enable_val_image_logging: bool = False,
        enable_inference_image_logging: bool = False,
        data_config: dict | None = None,
    ):
        """
        Initialize FNO model with neuralop backend.
        
        Args:
            n_modes: Number of Fourier modes to keep along each dimension (tuple of ints)
            hidden_channels: Width of the FNO layers
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_layers: Number of FNO layers
            non_linearity: Activation function
            stabilizer: Stabilization method ('tanh', None)
            norm: Normalization method ('ada_in', 'group_norm', 'instance_norm', None)
            preactivation: Whether to use pre-activation
            fno_skip: Type of skip connection for FNO ('linear', 'identity', 'soft-gating')
            separable: Whether to use separable convolutions
            factorization: Type of factorization ('tucker', 'tt', 'cp', None)
            rank: Rank for factorization (float for percentage, int for absolute)
            fixed_rank_modes: Whether to use fixed rank modes
            implementation: Implementation type ('factorized', 'reconstructed')
            decomposition_kwargs: Additional kwargs for tensor decomposition
            domain_padding: Amount of domain padding
            learning_rate: Learning rate for optimizer
            loss_function: Loss function name ("MSE", "MSE+Grad")
            num_predictions_to_log: Number of sequence predictions to visualize per validation epoch
            log_images_every_n_epochs: Log images only every Nth validation epoch
            max_image_logging_epochs: Global cap on how many epochs log images (None = no cap)
            enable_val_image_logging: Whether to log images during validation
            enable_inference_image_logging: Whether to log images during inference/predict
            data_config: Optional dict describing the data settings used for training
        """
        super().__init__(
            learning_rate=learning_rate,
            loss_function=loss_function,
            num_predictions_to_log=num_predictions_to_log,
            log_images_every_n_epochs=log_images_every_n_epochs,
            max_image_logging_epochs=max_image_logging_epochs,
            enable_val_image_logging=enable_val_image_logging,
            enable_inference_image_logging=enable_inference_image_logging,
        )
        
        self.save_hyperparameters()
        self.data_config = data_config
        
        fno_kwargs = {
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'n_layers': n_layers,
            'non_linearity': non_linearity,
            'preactivation': preactivation,
            'fno_skip': fno_skip,
            'separable': separable,
            'rank': rank,
            'fixed_rank_modes': fixed_rank_modes,
            'implementation': implementation,
        }
        
        if stabilizer is not None:
            fno_kwargs['stabilizer'] = stabilizer
        if norm is not None:
            fno_kwargs['norm'] = norm
        if factorization is not None:
            fno_kwargs['factorization'] = factorization
        if decomposition_kwargs is not None:
            fno_kwargs['decomposition_kwargs'] = decomposition_kwargs
        if domain_padding is not None:
            fno_kwargs['domain_padding'] = domain_padding
        
        self.fno = FNO(**fno_kwargs)
    
    def forward(self, x):
        """
        Forward pass through FNO.
        
        Supports inputs shaped either as:
            - [B, T, X, Y]  (single channel)
            - [B, T, C, X, Y] (multi-channel)
        Converts to: [B, C, X, Y, T] for neuralop FNO and reorders back afterward.
        """
        if x.ndim == 4:
            # [B, T, X, Y] -> [B, 1, X, Y, T]
            x = x.permute(0, 2, 3, 1).unsqueeze(1)
            multi_channel = False
        elif x.ndim == 5:
            # [B, T, C, X, Y] -> [B, C, X, Y, T]
            x = x.permute(0, 2, 3, 4, 1)
            multi_channel = True
        else:
            raise ValueError(f'Unsupported input shape {x.shape}; expected rank-4 or rank-5 tensor.')

        x = self.fno(x)

        if multi_channel:
            # [B, C_out, X, Y, T] -> [B, T, C_out, X, Y]
            x = x.permute(0, 4, 1, 2, 3)
        else:
            # [B, C_out=1, X, Y, T] -> [B, T, X, Y]
            x = x.squeeze(1).permute(0, 3, 1, 2)
        
        return x


class TFNOModel(BaseModel):
    """Temporal Fourier Neural Operator model using neuralop.TFNO.

    Expects inputs shaped [B, T, C, X, Y] and returns the same shape.
    """

    def __init__(
        self,
        n_modes: tuple = (16, 16, 16),
        hidden_channels: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        n_layers: int = 4,
        non_linearity = F.gelu,
        stabilizer: str | None = None,
        norm: str | None = None,
        preactivation: bool = False,
        fno_skip: str = 'linear',
        separable: bool = False,
        factorization: str | None = None,
        rank: float = 0.05,
        fixed_rank_modes: bool = False,
        implementation: str = 'factorized',
        decomposition_kwargs: dict | None = None,
        domain_padding: float | None = None,
        learning_rate: float = 1e-3,
        loss_function: str = "MSE",
        num_predictions_to_log: int = 1,
        log_images_every_n_epochs: int = 1,
        max_image_logging_epochs: int | None = None,
        enable_val_image_logging: bool = False,
        enable_inference_image_logging: bool = False,
        data_config: dict | None = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            loss_function=loss_function,
            num_predictions_to_log=num_predictions_to_log,
            log_images_every_n_epochs=log_images_every_n_epochs,
            max_image_logging_epochs=max_image_logging_epochs,
            enable_val_image_logging=enable_val_image_logging,
            enable_inference_image_logging=enable_inference_image_logging,
        )

        self.save_hyperparameters()
        self.data_config = data_config

        tfno_kwargs = {
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'n_layers': n_layers,
            'non_linearity': non_linearity,
            'preactivation': preactivation,
            'fno_skip': fno_skip,
            'separable': separable,
            'rank': rank,
            'fixed_rank_modes': fixed_rank_modes,
            'implementation': implementation,
        }

        if stabilizer is not None:
            tfno_kwargs['stabilizer'] = stabilizer
        if norm is not None:
            tfno_kwargs['norm'] = norm
        if factorization is not None:
            tfno_kwargs['factorization'] = factorization
        if decomposition_kwargs is not None:
            tfno_kwargs['decomposition_kwargs'] = decomposition_kwargs
        if domain_padding is not None:
            tfno_kwargs['domain_padding'] = domain_padding

        self.tfno = TFNO(**tfno_kwargs)

    def forward(self, x):
        # Same contract as FNOModel
        if x.ndim == 4:
            # [B, T, X, Y] -> [B, 1, X, Y, T]
            x = x.permute(0, 2, 3, 1).unsqueeze(1)
            multi_channel = False
        elif x.ndim == 5:
            # [B, T, C, X, Y] -> [B, C, X, Y, T]
            x = x.permute(0, 2, 3, 4, 1)
            multi_channel = True
        else:
            raise ValueError(f'Unsupported input shape {x.shape}; expected rank-4 or rank-5 tensor.')

        x = self.tfno(x)

        if multi_channel:
            # [B, C_out, X, Y, T] -> [B, T, C_out, X, Y]
            x = x.permute(0, 4, 1, 2, 3)
        else:
            # [B, C_out=1, X, Y, T] -> [B, T, X, Y]
            x = x.squeeze(1).permute(0, 3, 1, 2)

        return x


class DiffusionModel(BaseModel):
    def __init__(
        self,

        # UNet configurations
        dim,                                # Number of filters in the first layer. Used to initialize init_dim if init_dim is None
        init_dim = None,                    # The number of output channels for the initial convolution layer
        out_dim = None,                     # No of output channels
        dim_mults = (1, 2, 4, 8),           # Multipliers for the number of channels in each layer
        channels = 2,                       # Number of input channels
        self_condition = False,             # self-conditioning allows the model to use its own previous output as input alongside the current input during training
        learned_variance = False, 
        learned_sinusoidal_cond = False,
        random_fourier_features = False,    # Enables the use of random Fourier features for positioning in the latent space
        learned_sinusoidal_dim = 16, 
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,                 # Attention head dimensionality
        attn_heads = 4,                     # No. of attention heads
        flash_attn = False,   

        # GaussianDiffusion configurations
        image_size = 512,                   # Input image size
        timesteps = 1000,                   # No. of diffusion steps
        sampling_timesteps = 1000,          # The number of time steps to use during sampling.  Less than timesteps allows quicker generation
        objective = 'pred_noise',           # The training objective of the diffusion model.
        auto_normalize = False,              # Whether the input data should be automatically normalized
        min_snr_loss_weight = False,        # Whether to apply a minimum signal-to-noise ratio (SNR) weighting strategy to the loss function
        min_snr_gamma = 5,                  # Used in conjunction with min_snr_loss_weight to clamp the SNR values during training
        immiscible = False,

        # Training configurations
        learning_rate = 1e-3,
        loss_function = "MSE",
        num_predictions_to_log = 1,
        log_images_every_n_epochs = 1,
        max_image_logging_epochs = None,
        enable_val_image_logging = False,
        enable_inference_image_logging = False,

        data_min: float = float('inf'),
        data_max: float = float('-inf'),
    ):
        """
        Initialize Diffusion model with neuralop backend.
        """
        super().__init__(
            learning_rate=learning_rate,
            loss_function=loss_function,
            num_predictions_to_log=num_predictions_to_log,
            log_images_every_n_epochs=log_images_every_n_epochs,
            max_image_logging_epochs=max_image_logging_epochs,
            enable_val_image_logging=enable_val_image_logging,
            enable_inference_image_logging=enable_inference_image_logging,
        )
        
        self.save_hyperparameters()
        
        self.Unet = Unet(
            dim=dim,
            init_dim=init_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            channels=channels,
            self_condition=self_condition,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            sinusoidal_pos_emb_theta=sinusoidal_pos_emb_theta,
            dropout=dropout,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            flash_attn=flash_attn,
        )

        self.diffusion = GaussianDiffusion(
            model=self.Unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            auto_normalize=auto_normalize,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            immiscible=immiscible,
        )

        self.auto_normalize = auto_normalize
        self.data_min = data_min
        self.data_max = data_max
        self.timesteps = timesteps
        self.ddim_sampling_eta = 0.0

        self.loss_function = nn.MSELoss()

        print("data min and max:", data_min, data_max)
        print("auto normalize:", auto_normalize)
    def forward(self, x):
        """
        Forward pass through the denoising diffusion model.
        
        Supports inputs shaped as:
            - [B, C, X, Y]  (single channel)
        """

        if not self.auto_normalize:
            x = (x - self.data_min) / (self.data_max - self.data_min)
            x = x * 2.0 - 1.0

        x = self.diffusion(x) # returns the loss already
        
        return x
    
    @torch.inference_mode()
    def infer(self, x_start, steps=1000, eta=0.0):
        device = self.device
        diff = self.diffusion
        x = x_start.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [B, C, H, W] or [B, C, L]

        # Normalize to roughly [-1, 1] or whatever your training normalization was
        if not self.auto_normalize:
            x = (x - self.data_min) / (self.data_max - self.data_min)
        else:
            x = diff.normalize(x)  # assumes this does (x - data_min)/(data_max - data_min) * 2 - 1 or similar

        # === DDIM sampling (unchanged until the end) ===
        seq = torch.linspace(diff.num_timesteps-1, 0, steps+1, dtype=torch.long, device=device)
        seq_next = torch.cat([seq[1:], torch.tensor([-1], device=device)])

        x_t = x
        pred_x0_prev = None

        for i, (t_cur, t_next) in enumerate(zip(seq.tolist(), seq_next.tolist())):
            t_tensor = torch.full((x_t.shape[0],), t_cur, device=device, dtype=torch.long)

            pred_noise, pred_x0, *_ = diff.model_predictions(
                x_t,
                t_tensor,
                x_self_cond=pred_x0_prev if diff.self_condition else None,
                clip_x_start=True,
                rederive_pred_noise=True,
            )

            alpha_cur  = diff.alphas_cumprod[t_cur]
            alpha_next = diff.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(0.0, device=device)

            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next))

            direction = torch.sqrt(1 - alpha_next - sigma**2) * pred_noise
            noise = torch.randn_like(x_t)

            x_t = torch.sqrt(alpha_next) * pred_x0 + direction + sigma * noise

            pred_x0_prev = pred_x0

            if t_next < 0:
                x_t = pred_x0  # final deterministic step

        if not self.auto_normalize:
            # Reverse the [0,1] → original
            x_out = (x_t + 1.0) / 2.0                                  # → [0, 1]
            x_out = x_out * (self.data_max - self.data_min + 1e-8) + self.data_min
        else:
            x_out = diff.unnormalize(x_t)

        return x_out.squeeze(0)
        
    def training_step(self, batch, batch_idx):
        x, y = batch 
        x, y = x.to(self.device), y.to(self.device)
        train_loss = self(y)
        
        log_dict = {
            f"train_{self.loss_name}_loss": train_loss,
            "learning_rate": self.learning_rate
        }
        self.log_dict(
            log_dict, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True,
            logger=True  
        )
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        val_loss = self(x)
        
        self.log(
            f"val_{self.loss_name}_loss", 
            val_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        
        # Store outputs from first batch for end-of-epoch image logging
        # if self.enable_val_image_logging and batch_idx == 0:
        #     # Store up to num_predictions_to_log samples
        #     num_samples = min(self.num_predictions_to_log, x.shape[0])
        #     for i in range(num_samples):
        #         self.validation_step_outputs.append({
        #             'x': x[i:i+1].detach().cpu(),
        #             'y_true': y[i:i+1].detach().cpu(),
        #             'y_pred': y_hat[i:i+1].detach().cpu()
        #         })
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        test_loss = self(y)
        
        # Log metrics
        self.log(
            f"test_{self.loss_name}_loss", 
            test_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        
        return test_loss
    
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean = batch
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)  # optional ground truth

        # This uses your infer() → DDIM denoising from real noisy image
        denoised = self.infer(noisy, steps=2)  # Small number of steps for faster inference

        # Optional: log before/after images during prediction
        # if self.enable_inference_image_logging and batch_idx == 0 and self.logger is not None:
        #     num_samples = min(self.num_predictions_to_log, noisy.shape[0])
        #     for i in range(num_samples):
        #         self._log_predictions_as_images(
        #             x=noisy[i:i+1].cpu(),
        #             y_true=clean[i:i+1].cpu(),
        #             y_pred=denoised[i:i+1].cpu(),
        #             prefix='inference',
        #             sample_idx=i,
        #         )

        # Return denoised image + ground truth (for offline metrics if needed)
        return denoised, clean

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), logger=True)
    
    def on_validation_epoch_end(self):
        """Log images only on selected epochs to avoid flooding logs."""
        if not self.enable_val_image_logging:
            self.validation_step_outputs.clear()
            return

        should_log = False

        if self.max_image_logging_epochs is not None:
            if self._image_logging_epochs < self.max_image_logging_epochs:
                should_log = True
        else:
            should_log = True

        if should_log:
            if (int(self.current_epoch) % self.log_images_every_n_epochs) != 0:
                should_log = False

        if should_log and len(self.validation_step_outputs) > 0 and self.logger is not None:
            for idx, outputs in enumerate(self.validation_step_outputs):
                x = outputs['x']
                y_true = outputs['y_true']
                y_pred = outputs['y_pred']

                self._log_predictions_as_images(x, y_true, y_pred, prefix='val', sample_idx=idx)

            self._image_logging_epochs += 1
        
        # Clear stored outputs for next epoch
        self.validation_step_outputs.clear()
    
    def _log_predictions_as_images(self, x, y_true, y_pred, prefix='val', sample_idx=0):
        """
        Log predictions as images to TensorBoard or WandB.
        Each prediction consists of multiple timestep images (the full sequence length).
        
        Args:
            x: Input tensor [B, T, X, Y] where B=1 (single sample)
            y_true: Ground truth [B, T, X, Y]
            y_pred: Predictions [B, T, X, Y]
            prefix: Prefix for logging ('val' or 'test')
            sample_idx: Index of the sample being logged (for multiple predictions per epoch)
        """
        try:
            batch_idx = 0  # Always 0 since we store individual samples
            seq_len = y_true.shape[1]
            
            sample_true = y_true[batch_idx]
            sample_pred = y_pred[batch_idx]

            if sample_true.is_cuda:
                sample_true = sample_true.detach().cpu()
                sample_pred = sample_pred.detach().cpu()
            else:
                sample_true = sample_true.detach()
                sample_pred = sample_pred.detach()

            if sample_true.ndim == 3:
                y_true_np = sample_true.numpy()
                y_pred_np = sample_pred.numpy()
            elif sample_true.ndim == 4:
                # Multi-channel input; visualize the first channel by default.
                y_true_np = sample_true[:, 0].numpy()
                y_pred_np = sample_pred[:, 0].numpy()
            else:
                raise ValueError(
                    f'Unsupported tensor rank {sample_true.ndim} for image logging; expected 3 or 4 dimensions.'
                )
            
            timesteps_to_show = list(range(seq_len))
            num_timesteps = seq_len
            
            fig, axes = plt.subplots(3, num_timesteps, figsize=(4 * num_timesteps, 12))
            
            if num_timesteps == 1:
                axes = axes.reshape(-1, 1)
            
            for col_idx, t in enumerate(timesteps_to_show):
                true_frame = y_true_np[t]
                pred_frame = y_pred_np[t]
                error_frame = np.abs(true_frame - pred_frame)
                
                im0 = axes[0, col_idx].imshow(true_frame, cmap='viridis', aspect='auto')
                axes[0, col_idx].set_title(f'Ground Truth (t={t})')
                axes[0, col_idx].axis('off')
                plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046)
                
                im1 = axes[1, col_idx].imshow(pred_frame, cmap='viridis', aspect='auto')
                axes[1, col_idx].set_title(f'Prediction (t={t})')
                axes[1, col_idx].axis('off')
                plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046)
                
                im2 = axes[2, col_idx].imshow(error_frame, cmap='hot', aspect='auto')
                axes[2, col_idx].set_title(f'Abs Error (t={t})')
                axes[2, col_idx].axis('off')
                plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046)
            
            plt.tight_layout()
            
            # Use sample_idx in key name for multiple predictions
            key_suffix = f'_sample_{sample_idx}' if self.num_predictions_to_log > 1 else ''
            
            if isinstance(self.logger, TensorBoardLogger) and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(
                    f'{prefix}/predictions_sequence{key_suffix}',
                    fig,
                    global_step=self.current_epoch
                )
            elif isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    key=f'{prefix}/predictions_sequence{key_suffix}',
                    images=[fig],
                    step=self.current_epoch
                )
            
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log images: {e}")