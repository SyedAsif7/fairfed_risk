"""
FairFed Risk - Multi-Task PyTorch Dataset & DataLoader Utilities
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.partition import HospitalData


class ClinicalMultiTaskDataset(Dataset):
    """
    PyTorch Dataset providing:
      - x: (7,) float32 tensor of scaled clinical features
      - y: (3,) float32 tensor of disease risk indicators (diabetes, heart, ckd)
      - mask: (3,) float32 tensor indicating observed diagnoses
      - sensitive: (3,) int64 tensor [sex, age_group, hospital_id]
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        sensitive: np.ndarray,
    ):
        assert len(features) == len(targets) == len(masks) == len(sensitive)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.float32)
        self.sensitive = torch.as_tensor(sensitive, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.targets[idx],
            self.masks[idx],
            self.sensitive[idx],
        )


def build_dataloaders(
    hospital_data: HospitalData,
    batch_size: int = 32,
    shuffle_train: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Builds PyTorch DataLoaders for a specific hospital partition."""
    train_ds = ClinicalMultiTaskDataset(
        hospital_data.train_x,
        hospital_data.train_y,
        hospital_data.train_masks,
        hospital_data.train_sensitive,
    )
    test_ds = ClinicalMultiTaskDataset(
        hospital_data.test_x,
        hospital_data.test_y,
        hospital_data.test_masks,
        hospital_data.test_sensitive,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader
