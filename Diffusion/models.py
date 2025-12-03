import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

def get_meshgrid(shape):
    B, T, X, Y = shape[:4]
    t = torch.linspace(0, T - 1, T)
    x = torch.linspace(0, X - 1, X)
    y = torch.linspace(0, Y - 1, Y)

    tt, xx, yy = torch.meshgrid(t, x, y, indexing='ij')  # [T, X, Y]
    meshgrid = torch.stack([tt, xx, yy], dim=-1)         # [T, X, Y, 3]
    meshgrid = meshgrid.unsqueeze(0).expand(B, T, X, Y, 3)  # [B, T, X, Y, 3]
    return meshgrid

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class ConvBlock(nn.Module):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, kernel_size = 1, conv_dim = 3,activation='gelu'):
        super().__init__()

        # 1. Define conv layers
        self.conv1 = None
        self.conv2 = None
        if conv_dim == 3:
            self.conv1 = nn.Conv3d(in_neurons, hidden_neurons, kernel_size)
            self.conv2 = nn.Conv3d(hidden_neurons, out_neurons, kernel_size)
        elif conv_dim == 2:
            self.conv1 = nn.Conv2d(in_neurons, hidden_neurons, kernel_size)
            self.conv2 = nn.Conv2d(hidden_neurons, out_neurons, kernel_size)
        else:
            raise ValueError("dim must be either 2 or 3.")

        # 2. Define activation function
        self.activation = None
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
    
class FourierLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, modesSpace, modesTime, scaling=True):
        super().__init__()

        self.in_neurons = in_neurons    # From what I understand, fourier layer preserves input and output channels
        self.out_neurons = out_neurons  # From what I understand, fourier layer preserves input and output channels
        self.modesSpace = modesSpace
        self.modesTime = modesTime
        
        # Initialize weights
        if scaling:
            self.scale = 1 / (self.in_neurons * self.out_neurons)
        else:
            self.scale = 1
            
        self.weights  = nn.Parameter(self.scale * torch.rand(in_neurons, out_neurons, self.modesTime, self.modesSpace * 2, self.modesSpace * 2, dtype=torch.cfloat))

    def compl_mul3d(self, input): 
        # Batched matrix multiplication between the input tensor and the weights tensor
        output = torch.einsum('b o i x y, i o t x y -> b o t x y', input, self.weights)
        return output
        
    def forward(self, x):
        batchsize = x.shape[0]
        orig_time = x.shape[-3]
        orig_x = x.shape[-2]
        orig_y = x.shape[-1]

        # 1. FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1]) # FFT over time, and space     
        x_ft = torch.fft.fftshift(x_ft, dim=(-2, -1)) # Centering zero freq in the spatial spectrum

        # 2. Linear transform
        out_ft = torch.zeros(batchsize, self.out_neurons, self.modesTime, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x_ft.device) # device=x.device
        midX, midY =  x_ft.size(-2) // 2, x_ft.size(-1) // 2
        
        # Slicing: all batches and # of output neurons, limited to modesTime in time dimension, and modesSpace in spatial dimensions
        out_ft[..., :self.modesTime, midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace] = \
            self.compl_mul3d(x_ft[..., :self.modesTime, midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace])
        
        # 3. Inverse FFT
        out_ft = torch.fft.fftshift(out_ft, dim=(-2, -1))
        out_ft = torch.fft.irfftn(out_ft, s=(orig_time, orig_x, orig_y))
        return out_ft
    
class FNOModel(L.LightningModule):
    # def __init__(self, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, time_padding, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):
    def __init__(self, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):  
        super().__init__()

        # 1. Save hyperparameters
        self.learning_rate = learning_rate
        self.restart_at_epoch_n = restart_at_epoch_n
        # self.padding = time_padding # set padding here based on input_size
        self.n_batches = len(train_loader)
        self.n_training_samples = len(train_loader.dataset)
        self.loss_name = loss_function
        
        # 2. Network architechture definition
        self.p = nn.Linear(input_size, out_neurons)
        
        self.fourier1 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp1 = ConvBlock(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w1 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
 
        self.fourier2 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp2 = ConvBlock(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w2 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier3 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp3 = ConvBlock(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w3 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.fourier4 = FourierLayer(in_neurons, out_neurons, modesSpace, modesTime)
        self.mlp4 = ConvBlock(in_neurons, hidden_neurons, out_neurons, kernel_size=1)
        self.w4 = nn.Conv3d(in_neurons, out_neurons, kernel_size=1)
        
        self.q = ConvBlock(in_neurons, 4 * hidden_neurons, 1, kernel_size=1) # Single output predicts T timesteps
        
        if loss_function == 'L2':
            self.loss_function = LpLoss()
        elif loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
            
    def forward(self, x): # input dim: [B, T, X, Y]
        # 1. Add meshgrid (time and spacial coords info to input)
        meshgrid = get_meshgrid(x.shape).to(self.device) # Location & temporal information [B, T, X, Y, 3]
        x = x.unsqueeze(-1)  
        x = torch.concat((x, meshgrid), dim=-1) # [B, T, X, Y, O]
  
        # 2. Forward pass
        x = self.p(x) # [B, T, X, Y, O]
        x = x.permute(0, 1, 4, 2, 3)  # [B, T, O, X, Y]

        # x = F.pad(x, [0, 0, 0, 0, 0, 0, 0, self.padding]) # Zero-pad right of T dim
        x1 = self.fourier1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)

        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.fourier2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.fourier3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.fourier4(x)
        x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        
        # x = x[..., :-self.padding] # Unpad zeros
        x = self.q(x) 
        x = x.permute(0, 2, 3, 4, 1)  
        x = x.squeeze_(dim=-1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x) # [B, T, X, Y]
        train_loss = self.loss_function(y_hat, y)
        train_mse = F.mse_loss(y_hat, y)
        log_dict = {'mse_loss': train_mse, 'train_' + self.loss_name + '_loss': train_loss}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,T,X,Y]
        val_loss = self.loss_function(y_hat, y)
        self.log('val_' + self.loss_name + '_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,T,X,Y]
        test_loss = self.loss_function(y_hat, y)

        self.log('test_' + self.loss_name + '_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True)
        return test_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x), y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer