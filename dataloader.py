import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
from boutdata import collect
import torch.nn.functional as F

DEFAULT_DATA_ROOTS = {
    'data': Path('/dtu/blackhole/16/223702/data')
    # 'data': Path('/dtu/blackhole/1b/191611/data')
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
    channels: Tuple[str, ...] = ('n', 'phi') # ('n', 'pe', 'pi', 'te', 'ti')
    normalize: bool = True

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
        x = self.data[idx]   # [T, C, X, Y] or [T, X, Y]
        y = self.targets[idx]

        # Ensure we always have a channel dimension: [T, C, X, Y]
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [T, 1, X, Y]
            y = y.unsqueeze(1)

        # Downsample if spatial_resolution is specified
        if self.spatial_resolution is not None:
            T, C, X, Y = x.shape
            if X != self.spatial_resolution or Y != self.spatial_resolution:
                # [T, C, X, Y] -> [T*C, 1, X, Y] for interpolation
                x = x.view(T * C, 1, X, Y)
                y = y.view(T * C, 1, X, Y)

                x = F.interpolate(
                    x,
                    size=(self.spatial_resolution, self.spatial_resolution),
                    mode='bilinear',
                    align_corners=False,
                )
                y = F.interpolate(
                    y,
                    size=(self.spatial_resolution, self.spatial_resolution),
                    mode='bilinear',
                    align_corners=False,
                )

                # Back to [T, C, X, Y]
                x = x.view(T, C, self.spatial_resolution, self.spatial_resolution)
                y = y.view(T, C, self.spatial_resolution, self.spatial_resolution)

        return x, y

class DiffusionDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], targets: List[torch.Tensor], 
                 spatial_resolution: Optional[int] = None,  noise_mean: float = 0.0, noise_std: Union[float, Tuple[float, float]] = 0.02, 
                 ):
        """
        data: List of tensors representing the clean images.
        targets: List of tensors representing the target images (clean images).
        spatial_resolution: Target spatial resolution after cropping and resizing.

        noise_mean: Mean for Gaussian noise to be added.
        noise_std: Standard deviation for Gaussian noise to be added; Can be a float or a tuple (min, max) for random std.
        """

        self.data = data
        self.targets = targets
        self.spatial_resolution = spatial_resolution

        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]  # [C, X, Y]
        y = self.targets[idx]  # [C, X, Y]

        # 1. Resize images - per channel resizing
        if self.spatial_resolution is not None:
            # 1.1 Merge time and channel dims for interpolation then reshape back
            C, X, Y = x.shape
            x = x.reshape(C, 1, X, Y)
            y = y.reshape(C, 1, X, Y)

            # 1.2. Resize images
            x = F.interpolate(x, size=(self.spatial_resolution, self.spatial_resolution), 
                                mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(self.spatial_resolution, self.spatial_resolution), 
                                mode='bilinear', align_corners=False)

            # 1.3 Reshape back
            x = x.reshape(C, self.spatial_resolution, self.spatial_resolution)
            y = y.reshape(C, self.spatial_resolution, self.spatial_resolution)

        # 2 Add Gaussian noise to the resized images
        # 2.1 Get the std for noise
        if isinstance(self.noise_std, tuple):
            noise_std = np.random.uniform(self.noise_std[0], self.noise_std[1])            
        else:
            noise_std = self.noise_std

        noise = torch.normal(mean=self.noise_mean, std=noise_std, size=x.shape)
        x_noisy = x + noise

        return x_noisy, y

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
    channels: Tuple[str, ...] = DataLoaderConfig.channels,
    normalize: bool = DataLoaderConfig.normalize,
    dataset_name: str = 'PlasmaDataset',
    noise_mean: float = 0.0,
    noise_std: Union[float, Tuple[float, float]] = 0.02,
    
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    assert abs(train_split + val_split + test_split - 1.0) == 0, \
        f'Train, Val, Test splits must sum to 1.0 but are {train_split + val_split + test_split}'
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = [collect(var, path = str(data_root)) for var in channels]
    data_flattened = torch.from_numpy(np.stack([arr.squeeze() for arr in data], axis=1)).float()

    if normalize:
        reduce_dims = (0, 2, 3)
        mean = data_flattened.mean(dim=reduce_dims, keepdim=True)
        std = data_flattened.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        data_flattened = (data_flattened - mean) / std

    if dataset_name == 'PlasmaDataset':
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
    else:
        num_images = data_flattened.shape[0]
        train_end = int(num_images * train_split)
        val_end = int(num_images * (train_split + val_split))

        X_train = data_flattened[:train_end]
        X_val = data_flattened[train_end:val_end]
        X_test = data_flattened[val_end:]

        splits = [X_train, X_val, X_test]

        train_loader, val_loader, test_loader = [
            DataLoader(
                DiffusionDataset(X, X, spatial_resolution=spatial_resolution, noise_mean = noise_mean ,noise_std = noise_std),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            for X in splits
        ]
     
    print(
        "Data loaded: Original size, target resolution:"
        f" {spatial_resolution if spatial_resolution else 'original'}"
        f", normalization={'on' if normalize else 'off'}"
    )
    
    return train_loader, val_loader, test_loader