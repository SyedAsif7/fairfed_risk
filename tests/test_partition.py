"""
Unit tests for non-IID hospital partitioning and data leak prevention.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from src.data.partition import create_hospital_partitions, get_centralized_dataset
from src.data.dataset import build_dataloaders


@pytest.fixture(scope="module")
def partitions():
    return create_hospital_partitions(random_seed=42)


def test_partition_counts(partitions):
    """Verify 6 hospital partitions and total sample preservation."""
    assert len(partitions) == 6
    total_train = sum(p.n_train for p in partitions.values())
    total_test = sum(p.n_test for p in partitions.values())
    assert total_train + total_test == 2088
    # 80/20 train/test ratio check
    ratio = total_test / (total_train + total_test)
    assert 0.18 < ratio < 0.22


def test_no_nans_in_scaled_data(partitions):
    """Verify that train and test scaled feature matrices contain zero NaNs or infs."""
    for hid, p in partitions.items():
        assert not np.isnan(p.train_x).any(), f"NaNs found in Hospital {hid} train_x"
        assert not np.isnan(p.test_x).any(), f"NaNs found in Hospital {hid} test_x"
        assert not np.isinf(p.train_x).any(), f"Infs found in Hospital {hid} train_x"
        assert not np.isinf(p.test_x).any(), f"Infs found in Hospital {hid} test_x"


def test_feature_and_target_dimensions(partitions):
    """Verify standard shapes: 7 features, 3 targets, 3 masks."""
    for hid, p in partitions.items():
        assert p.train_x.shape[1] == 7
        assert p.train_y.shape[1] == 3
        assert p.train_masks.shape[1] == 3
        assert p.train_sensitive.shape[1] == 3
        assert p.test_x.shape[1] == 7
        assert p.test_y.shape[1] == 3


def test_centralized_dataset_assembly(partitions):
    """Verify pooled centralized dataset shapes and consistency."""
    tr_x, tr_y, tr_m, tr_s, te_x, te_y, te_m, te_s = get_centralized_dataset(partitions)
    assert tr_x.shape[0] == sum(p.n_train for p in partitions.values())
    assert te_x.shape[0] == sum(p.n_test for p in partitions.values())
    assert tr_x.shape[1] == 7
    assert tr_y.shape[1] == 3


def test_dataloaders(partitions):
    """Verify PyTorch DataLoader yields correct tensor types and dimensions."""
    h0 = partitions[0]
    train_loader, test_loader = build_dataloaders(h0, batch_size=16)
    x, y, mask, sensitive = next(iter(train_loader))
    assert isinstance(x, torch.Tensor)
    assert x.shape == (16, 7)
    assert y.shape == (16, 3)
    assert mask.shape == (16, 3)
    assert sensitive.shape == (16, 3)
