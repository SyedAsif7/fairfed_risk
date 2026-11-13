"""
Unit tests for data download and harmonization modules.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from src.data.harmonize import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    MASK_COLUMNS,
    load_harmonized_dataset,
)


@pytest.fixture(scope="module")
def harmonized_df():
    return load_harmonized_dataset(raw_dir=Path("data/raw"))


def test_harmonized_shape_and_columns(harmonized_df):
    """Verify total row count and required schema columns."""
    assert len(harmonized_df) == 2088
    for col in FEATURE_COLUMNS + TARGET_COLUMNS + MASK_COLUMNS:
        assert col in harmonized_df.columns, f"Missing expected column: {col}"


def test_target_and_mask_binary(harmonized_df):
    """Verify targets and masks are strictly binary."""
    for col in TARGET_COLUMNS + MASK_COLUMNS:
        unique_vals = set(harmonized_df[col].dropna().unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary values in {col}: {unique_vals}"


def test_mask_target_consistency(harmonized_df):
    """Verify that if mask is 1, target is observed, and positive prevalence is plausible."""
    for t_col, m_col in zip(TARGET_COLUMNS, MASK_COLUMNS):
        observed = harmonized_df[m_col] == 1
        assert observed.sum() > 0, f"No observed labels for {t_col}"
        pos_rate = (harmonized_df[t_col] * harmonized_df[m_col]).sum() / observed.sum()
        assert 0.05 < pos_rate < 0.90, f"Plausible prevalence for {t_col}: {pos_rate}"


def test_demographic_attributes(harmonized_df):
    """Verify sensitive demographic attributes exist and have expected categories."""
    assert "sex" in harmonized_df.columns
    assert set(harmonized_df["sex"].unique()).issubset({0.0, 1.0})
    assert "age_group" in harmonized_df.columns
    assert set(harmonized_df["age_group"].unique()).issubset({0, 1, 2})
