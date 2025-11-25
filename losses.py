import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.losses import H1Loss, LpLoss


class Neuralop_LpLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self._loss = LpLoss(d=2, p=2, reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self._loss is not None:
            # neuraloperator LpLoss typically expects (pred, target)
            return self._loss(y_pred, y_true)
        return F.mse_loss(y_pred, y_true)


class Neuralop_H1Loss(nn.Module):

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self._loss = H1Loss(2)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._loss(y_pred, y_true)



class GradientLoss(nn.Module):

    def __init__(self, loss_type="L2"):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, y_pred, y_true):
        dx_pred = y_pred[..., 1:, :] - y_pred[..., :-1, :]
        dy_pred = y_pred[..., :, 1:] - y_pred[..., :, :-1]

        dx_true = y_true[..., 1:, :] - y_true[..., :-1, :]
        dy_true = y_true[..., :, 1:] - y_true[..., :, :-1]

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

__all__ = ["Neuralop_LpLoss", "Neuralop_H1Loss", "GradientLoss", "MSEWithGradientLoss"]