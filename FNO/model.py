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
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure


class BaseModel(L.LightningModule):
    """Base class for all models with shared training/validation/test logic."""
    
    def __init__(self, learning_rate: float, loss_function: str, num_predictions_to_log: int = 1, log_images_every_n_epochs: int = 1, max_image_logging_epochs: int | None = None, enable_val_image_logging: bool = False, enable_inference_image_logging: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_name = loss_function
        self.num_predictions_to_log = num_predictions_to_log
        self.log_images_every_n_epochs = max(1, int(log_images_every_n_epochs))
        self.max_image_logging_epochs = max_image_logging_epochs
        self._image_logging_epochs = 0
        self.enable_val_image_logging = enable_val_image_logging
        self.enable_inference_image_logging = enable_inference_image_logging
        
        self.validation_step_outputs = []

        self.ssim = StructuralSimilarityIndexMeasure()
        self.mssim = MultiScaleStructuralSimilarityIndexMeasure(betas=(0.0448, 0.2856, 0.3001), kernel_size=7)

        if loss_function in ("MSE", "L2", "LpLoss"):
            self.loss_function = Neuralop_LpLoss()
        elif loss_function == "MSE+Grad":
            self.loss_function = MSEWithGradientLoss()
        elif loss_function == "H1":
            self.loss_function = Neuralop_H1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    def _compute_loss(self, y_hat, y, inputs=None):
        """Call loss function, passing `inputs` when accepted by the loss.

        This helper tries to call the configured loss with (pred, true, inputs)
        and falls back to (pred, true) when the loss doesn't expect an inputs
        argument.
        """
        if self.loss_function is None:
            raise RuntimeError('Loss function not configured; ensure model initializes LossMHD with correct data_config or choose a supported loss.')
        try:
            return self.loss_function(y_hat, y, inputs)
        except TypeError:
            return self.loss_function(y_hat, y)
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        train_loss = self._compute_loss(y_hat, y, x)
        
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
        val_loss = self._compute_loss(y_hat, y, x)
        
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
        test_loss = self._compute_loss(y_hat, y, x)
        self.log(f"test_{self.loss_name}_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        # Compute additional metrics
        ssim_scores = []
        mssim_scores = []
        for t in range(y_hat.shape[1]):
            ssim_scores.append(self.ssim(y_hat[:, t], y[:, t]))
            mssim_scores.append(self.mssim(y_hat[:, t], y[:, t]))

        ssim = torch.stack(ssim_scores).mean()
        mssim = torch.stack(mssim_scores).mean()

        self.log("test_ssim", ssim, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log("test_mssim", mssim, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        
        return test_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)

        if self.enable_inference_image_logging and self.logger is not None:
            sample_to_show = getattr(self, 'sample_to_show', None)
            batch_size = int(x.shape[0])
            logged = False

            if sample_to_show is not None:
                try:
                    sample_global_idx = int(sample_to_show)
                except Exception:
                    sample_global_idx = None

                if sample_global_idx is not None:
                    batch_start = batch_idx * batch_size
                    batch_end = batch_start + batch_size  # exclusive
                    if batch_start <= sample_global_idx < batch_end:
                        local_i = sample_global_idx - batch_start
                        self._log_predictions_as_images(
                            x[local_i:local_i+1].detach().cpu(),
                            y[local_i:local_i+1].detach().cpu(),
                            y_hat[local_i:local_i+1].detach().cpu(),
                            prefix='inference',
                            sample_idx=sample_global_idx,
                        )
                        logged = True

            if not logged:
                if batch_idx == 0:
                    num_samples = min(self.num_predictions_to_log, batch_size)
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
        
        self.validation_step_outputs.clear()
    
    def _log_predictions_as_images(self, x, y_true, y_pred, prefix='val', sample_idx=0, timesteps_to_show=None):
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
            batch_idx = 0  
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
                y_true_np = sample_true[:, 0].numpy()
                y_pred_np = sample_pred[:, 0].numpy()
            else:
                raise ValueError(
                    f'Unsupported tensor rank {sample_true.ndim} for image logging; expected 3 or 4 dimensions.'
                )
            if timesteps_to_show is None:
                timesteps_to_show = list(range(seq_len))
            else:
                if isinstance(timesteps_to_show, int):
                    timesteps_to_show = [int(timesteps_to_show)]
                elif isinstance(timesteps_to_show, str):
                    tts_str = timesteps_to_show.lower()
                    if tts_str in ('all', 'none'):
                        timesteps_to_show = list(range(seq_len))
                    elif tts_str == 'last':
                        timesteps_to_show = [seq_len - 1]
                    elif tts_str == 'random':
                        low = max(1, seq_len // 2)
                        high = seq_len
                        if high <= low:
                            t = seq_len - 1
                        else:
                            t = int(np.random.randint(low=low, high=high))
                        timesteps_to_show = [t]
                    elif ',' in timesteps_to_show:
                        timesteps_to_show = [int(s) for s in timesteps_to_show.split(',')]
                    else:
                        try:
                            timesteps_to_show = [int(timesteps_to_show)]
                        except Exception:
                            timesteps_to_show = list(range(seq_len))
                else:
                    timesteps_to_show = [int(t) for t in timesteps_to_show]
            clamped_timesteps = []
            invalid = []
            for t in timesteps_to_show:
                if t < 0 or t >= seq_len:
                    invalid.append(t)
                    clamped_timesteps.append(max(0, min(seq_len - 1, int(t))))
                else:
                    clamped_timesteps.append(int(t))

            if invalid:
                msg = (
                    f'Requested timesteps {invalid} out of range [0, {seq_len - 1}]; '
                    f'clamped to {clamped_timesteps}.'
                )
                try:
                    if isinstance(self.logger, WandbLogger) and hasattr(self.logger, 'experiment'):
                        print(msg)
                        try:
                            self.logger.experiment.log({f"{prefix}_timestep_clamp": msg}, step=self.current_epoch)
                        except Exception:
                            pass
                    else:
                        print(msg)
                except Exception:
                    print(msg)

            timesteps_to_show = clamped_timesteps
            num_timesteps = len(timesteps_to_show)

            if seq_len == 1:
                fig, axes = plt.subplots(3, 1, figsize=(6, 12))
                if axes.ndim == 1:
                    axes = axes.reshape(3, 1)
            else:
                cbar_width = 0.05
                spacer_width = 0.08
                fig_width = max(4 * num_timesteps, 6)
                fig = plt.figure(figsize=(fig_width + 2.5, 12))
                gs = fig.add_gridspec(
                    nrows=3,
                    ncols=num_timesteps + 3,
                    width_ratios=[1] * num_timesteps + [cbar_width, spacer_width, cbar_width],
                    wspace=0.05,
                    hspace=0.12,
                )
                axes = np.empty((3, num_timesteps), dtype=object)
                for r in range(3):
                    for c in range(num_timesteps):
                        axes[r, c] = fig.add_subplot(gs[r, c])
                cbar_ax_val = fig.add_subplot(gs[:, num_timesteps])
                cbar_ax_err = fig.add_subplot(gs[:, num_timesteps + 2])
            
            error_stack = np.abs(y_true_np[timesteps_to_show] - y_pred_np[timesteps_to_show])
            global_vmin = 0.0
            global_vmax = float(error_stack.max()) if error_stack.size > 0 else 1.0

            for col_idx, t in enumerate(timesteps_to_show):
                true_frame = y_true_np[t]
                pred_frame = y_pred_np[t]
                error_frame = np.abs(true_frame - pred_frame)

                im0 = axes[0, col_idx].imshow(true_frame, cmap='viridis', aspect='auto')
                axes[0, col_idx].set_title(f'Ground Truth (t={t})')
                axes[0, col_idx].axis('off')

                im1 = axes[1, col_idx].imshow(pred_frame, cmap='viridis', aspect='auto')
                axes[1, col_idx].set_title(f'Prediction (t={t})')
                axes[1, col_idx].axis('off')

                im2 = axes[2, col_idx].imshow(error_frame, cmap='hot', aspect='auto', vmin=global_vmin, vmax=global_vmax)
                axes[2, col_idx].set_title(f'Abs Error (t={t})')
                axes[2, col_idx].axis('off')

            try:
                val_stack = np.concatenate([y_true_np[timesteps_to_show], y_pred_np[timesteps_to_show]], axis=0)
                val_vmin = float(val_stack.min()) if val_stack.size > 0 else float(np.min(y_true_np))
                val_vmax = float(val_stack.max()) if val_stack.size > 0 else float(np.max(y_true_np))
                mappable_val = plt.cm.ScalarMappable(cmap='viridis')
                mappable_val.set_clim(val_vmin, val_vmax)
                if seq_len == 1:
                    cbar_gt = plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
                    cbar_pred = plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                    try:
                        cbar_gt.ax.tick_params(labelsize=8)
                        cbar_pred.ax.tick_params(labelsize=8)
                    except Exception:
                        pass
                else:
                    cbar_val = fig.colorbar(mappable_val, cax=cbar_ax_val, fraction=0.02, pad=0.02)
                    cbar_val.ax.set_title('Value')
                    cbar_val.ax.tick_params(labelsize=8)
                    try:
                        cbar_val.ax.title.set_fontsize(10)
                    except Exception:
                        pass

                mappable_err = plt.cm.ScalarMappable(cmap='hot')
                mappable_err.set_clim(global_vmin, global_vmax)
                if seq_len == 1:
                    cbar_err = plt.colorbar(im2, ax=axes[2, 0], fraction=0.046)
                    try:
                        cbar_err.ax.set_title('Abs Error')
                        cbar_err.ax.tick_params(labelsize=8)
                        cbar_err.ax.title.set_fontsize(10)
                    except Exception:
                        pass
                else:
                    cbar_err = fig.colorbar(mappable_err, cax=cbar_ax_err, fraction=0.02, pad=0.02)
                    cbar_err.ax.set_title('Abs Error')
                    cbar_err.ax.tick_params(labelsize=8)
                    try:
                        cbar_err.ax.title.set_fontsize(10)
                    except Exception:
                        pass
            except Exception:
                for col_idx in range(num_timesteps):
                    true_frame = y_true_np[col_idx]
                    pred_frame = y_pred_np[col_idx]
                    error_frame = np.abs(true_frame - pred_frame)
                    im0 = axes[0, col_idx].imshow(true_frame, cmap='viridis', aspect='auto')
                    plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046)
                    im1 = axes[1, col_idx].imshow(pred_frame, cmap='viridis', aspect='auto')
                    plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046)
                    im2 = axes[2, col_idx].imshow(error_frame, cmap='hot', aspect='auto')
                    plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046)
            
            plt.tight_layout()
            
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