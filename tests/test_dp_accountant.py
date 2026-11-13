"""
Unit tests for Differential Privacy Accountant and Opacus DP-SGD integration.
"""

from __future__ import annotations
import numpy as np
import pytest

from src.privacy.accountant import PrivacyAccountant
from src.federated.client import HospitalFlowerClient
from src.data.partition import create_hospital_partitions


def test_accountant_monotonic_growth():
    """Verify cumulative epsilon monotonically increases as training steps advance."""
    acc = PrivacyAccountant(delta=1e-5)
    eps_history = []
    for _ in range(5):
        acc.step(noise_multiplier=1.5, sample_rate=0.1)
        eps_history.append(acc.get_epsilon())

    for i in range(1, len(eps_history)):
        assert eps_history[i] > eps_history[i - 1], "Epsilon did not increase after DP step"


def test_accountant_noise_scaling():
    """Verify higher noise multiplier produces strictly smaller epsilon (higher privacy)."""
    acc_low_noise = PrivacyAccountant(delta=1e-5)
    acc_high_noise = PrivacyAccountant(delta=1e-5)

    for _ in range(50):
        acc_low_noise.step(noise_multiplier=0.8, sample_rate=0.1)
        acc_high_noise.step(noise_multiplier=3.0, sample_rate=0.1)

    eps_low = acc_low_noise.get_epsilon()
    eps_high = acc_high_noise.get_epsilon()

    assert eps_high < eps_low, f"Expected eps_high ({eps_high}) < eps_low ({eps_low})"


def test_accountant_calibration():
    """Verify calibrated noise multiplier produces targeted epsilon values."""
    target_eps = 3.0
    calibrated_sigma = PrivacyAccountant.calibrate_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=1e-5,
        sample_rate=0.12,
        total_steps=100,
    )
    assert calibrated_sigma > 0.5

    # Test that calibrated sigma results in target_eps (+/- 0.05)
    acc = PrivacyAccountant(delta=1e-5)
    for _ in range(100):
        acc.step(noise_multiplier=calibrated_sigma, sample_rate=0.12)
    computed_eps = acc.get_epsilon()

    assert abs(computed_eps - target_eps) < 0.05, f"Calibration error: {computed_eps} vs {target_eps}"


def test_accountant_infinity_target():
    """Verify non-private baseline (epsilon=inf) returns sigma=0."""
    sigma = PrivacyAccountant.calibrate_noise_multiplier(float("inf"))
    assert sigma == 0.0


def test_dp_client_fit():
    """Verify HospitalFlowerClient operates under DP-SGD and tracks epsilon."""
    partitions = create_hospital_partitions(random_seed=42)
    h0 = partitions[0]

    client = HospitalFlowerClient(
        hospital_data=h0,
        batch_size=32,
        lr=0.005,
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        delta=1e-5,
    )
    init_params = client.get_parameters(config={})
    updated_params, n_samples, fit_res = client.fit(init_params, config={"local_epochs": 1})

    assert len(updated_params) == len(init_params)
    assert "epsilon_spent" in fit_res
    assert 0.0 < fit_res["epsilon_spent"] < 100.0
