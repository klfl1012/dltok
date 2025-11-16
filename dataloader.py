import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import numpy as np
from boutdata import collect 

DEFAULT_DATA_ROOTS = {
    'data': Path('/data/')
}

@dataclass(frozen=True)
class DataLoaderConfig:
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 21
    shuffle: bool = False
    data_root: Path = DEFAULT_DATA_ROOTS['data']


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
    def __init__(self, data: List[torch.Tensor], targets: List[torch.Tensor]):
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


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
            PlasmaDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for X, y in splits
    ]
    
    return train_loader, val_loader, test_loader