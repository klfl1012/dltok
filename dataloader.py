import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from boutdata import collect
import torch.nn.functional as F

DEFAULT_DATA_ROOTS = {
    'data': Path('/dtu/blackhole/16/223702/data')
}

@dataclass(frozen=True)
class DataLoaderConfig:
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    num_workers: int = 0  # Changed from 4 to 0 for CPU RAM loading
    pin_memory: bool = False  # Changed from True to False to keep data in CPU RAM
    seed: int = 21
    shuffle: bool = False
    data_root: Path = DEFAULT_DATA_ROOTS['data']
    spatial_resolution: Optional[int] = 256  # Target resolution (default: downsample 1024->256)


def _tensor_to_sequences(
        data_tensor: torch.Tensor, 
        sequence_length: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data_sequences = []
    target_sequences = []
    for i in range(len(data_tensor) - sequence_length):
        data_sequences.append(data_tensor[i:i + sequence_length]) 
        target_sequences.append(data_tensor[i + 1 : i + sequence_length + 1]) 
    
    return data_sequences, target_sequences


class PlasmaDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], targets: List[torch.Tensor], spatial_resolution: Optional[int] = None):
        self.data = data
        self.targets = targets
        self.spatial_resolution = spatial_resolution

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]  # [T, X, Y]
        y = self.targets[idx]  # [T, X, Y]
        
        # Downsample if spatial_resolution is specified
        if self.spatial_resolution is not None:
            T, X, Y = x.shape
            if X != self.spatial_resolution or Y != self.spatial_resolution:
                # Reshape to [T, 1, X, Y] for interpolation
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                
                # Downsample using bilinear interpolation
                x = F.interpolate(x, size=(self.spatial_resolution, self.spatial_resolution), 
                                 mode='bilinear', align_corners=False)
                y = F.interpolate(y, size=(self.spatial_resolution, self.spatial_resolution), 
                                 mode='bilinear', align_corners=False)
                
                # Reshape back to [T, X, Y]
                x = x.squeeze(1)
                y = y.squeeze(1)
        
        return x, y


def build_dataloader(
    seq_len: int,
    train_split: float = DataLoaderConfig.train_split,
    val_split: float = DataLoaderConfig.val_split,
    test_split: float = DataLoaderConfig.test_split,
    batch_size: int = DataLoaderConfig.batch_size,
    num_workers: int = DataLoaderConfig.num_workers,
    pin_memory: bool = DataLoaderConfig.pin_memory,
    seed: int = DataLoaderConfig.seed,
    shuffle: bool = DataLoaderConfig.shuffle,
    data_root: Path = DataLoaderConfig.data_root,
    spatial_resolution: Optional[int] = DataLoaderConfig.spatial_resolution,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    assert abs(train_split + val_split + test_split - 1.0) == 0, \
        f'Train, Val, Test splits must sum to 1.0 but are {train_split + val_split + test_split}'
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = collect('n', path = str(data_root))
    data_flattened = torch.from_numpy(data.squeeze()).float()

    X_seq, y_seq = _tensor_to_sequences(data_flattened, seq_len)
    
    total_sequences = len(X_seq)
    train_end = int(total_sequences * train_split)
    val_end = int(total_sequences * (train_split + val_split))
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    splits = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    train_loader, val_loader, test_loader = [
        DataLoader(
            PlasmaDataset(X, y, spatial_resolution=spatial_resolution),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for X, y in splits
    ]
    
    print(f"Data loaded: Original size, target resolution: {spatial_resolution if spatial_resolution else 'original'}")
    
    return train_loader, val_loader, test_loader