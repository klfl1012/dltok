import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from abc import abstractmethod
from neuralop import FNO
import matplotlib.pyplot as plt
import numpy as np


class GradientLoss(nn.Module):

    def __init__(self, loss_type="L2"):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, y_pred, y_true):
        dx_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        dy_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        dx_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        dy_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]

        loss = F.mse_loss(dx_pred, dx_true) + F.mse_loss(dy_pred, dy_true)

        return loss

class MSEWithGradientLoss(nn.Module):

    def __init__(self, alpha: float=0.5):
        super(MSEWithGradientLoss, self).__init__()
        self.alpha = alpha
        self.gradient_loss = GradientLoss()

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        grad_loss = self.gradient_loss(y_pred, y_true)
        loss = mse_loss + self.alpha * grad_loss

        return loss


class BaseModel(L.LightningModule):
    """Base class for all models with shared training/validation/test logic."""
    
    def __init__(self, learning_rate: float, loss_function: str):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_name = loss_function
        self.num_images_to_log = 1  # Default, can be overridden
        
        # Loss function setup
        if loss_function == "MSE":
            self.loss_function = F.mse_loss
        elif loss_function == "MSE+Grad":
            self.loss_function = MSEWithGradientLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        train_loss = self.loss_function(y_hat, y)
        
        # Log metrics
        log_dict = {
            f"train_{self.loss_name}_loss": train_loss,
            "learning_rate": self.learning_rate
        }
        self.log_dict(
            log_dict, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True,
            logger=True  # Ensures TensorBoard logging
        )
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)
        
        # Log metrics
        self.log(
            f"val_{self.loss_name}_loss", 
            val_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        
        # Log prediction images (only first batch of each epoch)
        if batch_idx == 0 and self.logger is not None:
            self._log_predictions_as_images(x, y, y_hat, prefix='val')
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
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
        return self(x), y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        # Log current epoch number
        self.log("epoch", float(self.current_epoch), logger=True)
    
    def on_validation_epoch_end(self):
        pass
    
    def _log_predictions_as_images(self, x, y_true, y_pred, prefix='val'):
        """
        Log predictions as images to TensorBoard.
        
        Args:
            x: Input tensor [B, T, X, Y]
            y_true: Ground truth [B, T, X, Y]
            y_pred: Predictions [B, T, X, Y]
            prefix: Prefix for logging ('val' or 'test')
        """
        try:
            num_samples = min(self.num_images_to_log, x.shape[0])
            
            for i in range(num_samples):
                # Take middle timestep for visualization
                mid_t = y_true.shape[1] // 2
                
                true_frame = y_true[i, mid_t].detach().cpu().numpy()
                pred_frame = y_pred[i, mid_t].detach().cpu().numpy()
                
                # Create comparison plot with fixed compact size
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Ground truth
                im0 = axes[0].imshow(true_frame, cmap='viridis', aspect='auto')
                axes[0].set_title(f'Ground Truth (t={mid_t})')
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], fraction=0.046)
                
                # Prediction
                im1 = axes[1].imshow(pred_frame, cmap='viridis', aspect='auto')
                axes[1].set_title(f'Prediction (t={mid_t})')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                
                # Absolute error
                diff = np.abs(true_frame - pred_frame)
                im2 = axes[2].imshow(diff, cmap='hot', aspect='auto')
                axes[2].set_title(f'Absolute Error')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                
                plt.tight_layout()
                
                # Log to TensorBoard
                if isinstance(self.logger, TensorBoardLogger) and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_figure(
                        f'{prefix}/predictions_sample_{i}',
                        fig,
                        global_step=self.current_epoch
                    )
                
                plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log images to TensorBoard: {e}")





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
        """
        super().__init__(learning_rate=learning_rate, loss_function=loss_function)
        
        self.save_hyperparameters()
        
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
        
        Expects input shape: [B, T, X, Y] 
        Converts to: [B, C=1, X, Y, T] for neuralop FNO
        Returns: [B, 1, X, Y, T] -> converted back to [B, T, X, Y]
        """
        # Input: [B, T, X, Y]
        if x.ndim == 4:
            # Permute to [B, X, Y, T] and add channel dimension -> [B, 1, X, Y, T]
            x = x.permute(0, 2, 3, 1).unsqueeze(1)
        
        # FNO forward pass: [B, 1, X, Y, T] -> [B, 1, X, Y, T]
        x = self.fno(x)
        
        # Convert back: [B, 1, X, Y, T] -> [B, T, X, Y]
        x = x.squeeze(1).permute(0, 3, 1, 2)
        
        return x