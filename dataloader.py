import torch
from torch.utils.data import Dataset, DataLoader

def tensor_to_sequences(data_tensor, sequence_length):
    data_sequences = []
    target_sequences = []
    for i in range(len(data_tensor) - sequence_length):
        data_sequences.append(data_tensor[i:i + sequence_length]) 
        target_sequences.append(data_tensor[i + 1 : i + sequence_length + 1]) 
    
    return data_sequences, target_sequences

def bout_array_to_sequences(bout_array, sequence_length):
    # 1. Convert bout_array to a PyTorch FloatTensor
    data_tensor = torch.FloatTensor(bout_array.view().reshape(bout_array.shape))

    # 2. Create sequences and targets sliding window 
    return tensor_to_sequences(data_tensor, sequence_length)

class PlasmaDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]