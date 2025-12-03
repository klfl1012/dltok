from matplotlib import pyplot as plt
from boutdata import collect
import torch
import lightning as L

import models
import dataloader
from torch.utils.data import DataLoader

# 1. Read data
n = torch.load('/dtu/blackhole/1b/191611/Data/data_tensor.pt')

# 2. Put data in dataloader
seq_length = 3
X_seq, y_seq = dataloader.tensor_to_sequences(n, sequence_length = seq_length) 

batch_size = 2 
dataset = dataloader.PlasmaDataset(X_seq, y_seq)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model parameters
in_neurons = seq_length  # Input neurons equals number of timesteps for your sequences
hidden_neurons = 64      
out_neurons = seq_length  
# out_neurons = 1 

modesSpace = 16  # No of wave patterns for spatial dimensions
modesTime = 16  # No of wave patterns for time dimension

# time_padding = 1  # Padding for the convolution
input_size = 4  # (1 for t, 1 for x, 1 for y) + 1

learning_rate = 1e-4
restart_at_epoch_n = 10  # Number of epochs before restarting the learning rate scheduler
loss_function = 'MSE'  # Choose your desired loss function

# Initialize the model
# model = models.FNOModel(in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, time_padding, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function)
model = models.FNOModel(in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function)
# Initialize and train with PyTorch Lightning
trainer = L.Trainer(
    max_epochs=100
)

# Train
trainer.fit(model, train_loader)