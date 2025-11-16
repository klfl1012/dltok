import torch
import torch.nn.functional as F
import lightning as L
from abc import abstractmethod


class LpLoss(object):
    """Lp loss function for relative error computation."""
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0
        
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    
    def abs(self, x, y):
        num_examples = x.size()[0]
        
        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        
        all_norms = (h**(self.d/self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        
        return all_norms
    
    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        
        return diff_norms / y_norms
    
    def __call__(self, x, y):
        return self.rel(x, y)


class BaseModel(L.LightningModule):
    """Base class for all models with shared training/validation/test logic."""
    
    def __init__(self, learning_rate: float, loss_function: str):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_name = loss_function
        
        # Loss function setup
        if loss_function == 'L2':
            self.loss_function = LpLoss()
        elif loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        train_loss = self.loss_function(y_hat, y)
        train_mse = F.mse_loss(y_hat, y)
        
        # Log metrics
        log_dict = {
            'mse_loss': train_mse, 
            f'train_{self.loss_name}_loss': train_loss,
            'learning_rate': self.learning_rate
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
        """
        Validation step with automatic logging.
        Logs to both TensorBoard and progress bar.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)
        val_mse = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.log(
            f'val_{self.loss_name}_loss', 
            val_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        self.log(
            'val_mse_loss',
            val_mse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True
        )
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        test_mse = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.log(
            f'test_{self.loss_name}_loss', 
            test_loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True,
            logger=True
        )
        self.log(
            'test_mse_loss',
            test_mse,
            prog_bar=False,
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
        self.log('epoch', float(self.current_epoch), logger=True)
    
    def on_validation_epoch_end(self):
        pass
