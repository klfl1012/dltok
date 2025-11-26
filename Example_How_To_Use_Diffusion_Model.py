from matplotlib import pyplot as plt
from boutdata import collect
import torch
import lightning as L

from model import DiffusionModel

import dataloader
from torch.utils.data import DataLoader

checkpoint_path = "ckpts/diffusion_res256_seq100_20251126_212322-best.ckpt"

spatial_resolution = 256

batch_size = 8 
train_loader, val_loader, test_loader = dataloader.build_dataloader(
    seq_len=4,
    batch_size=batch_size,
    seed=12,
    spatial_resolution=spatial_resolution,
    normalize=True,
    dataset_name='Test',
)

model = DiffusionModel.load_from_checkpoint(checkpoint_path)


input_batch = next(iter(test_loader))  # Get a single batch
x = input_batch[0].to(model.device)  # Input tensor
y = input_batch[1].to(model.device)  # Target tensor

y_hat = model.infer(x)

# Move tensors to CPU for visualization and detach them
x_cpu = x.cpu().detach()
y_cpu = y.cpu().detach()
y_hat_cpu = y_hat.cpu().detach()

print(f"Input shape: {x_cpu.shape}")
print(f"Target shape: {y_cpu.shape}")
print(f"Prediction shape: {y_hat_cpu.shape}")

# Export all 3 tensors 
torch.save(x_cpu, "x_tensor.pt")
torch.save(y_cpu, "y_tensor.pt")
torch.save(y_hat_cpu, "y_hat_tensor.pt")
