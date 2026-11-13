"""
Unit tests for Federated Learning components: aggregation, clients, and proximal regularization.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from src.federated.strategies import aggregate_weights, set_model_weights_from_ndarrays
from src.federated.client import HospitalFlowerClient
from src.federated.server import FederatedSimulationServer
from src.data.partition import create_hospital_partitions
from src.models.multitask_mlp import MultiTaskRiskMLP


def test_aggregate_weights_exactness():
    """Verify weighted averaging behaves exactly as linear combination."""
    arr1 = [np.array([1.0, 2.0, 3.0]), np.array([[10.0, 20.0]])]
    arr2 = [np.array([3.0, 6.0, 9.0]), np.array([[30.0, 40.0]])]

    # Equal weights 0.5, 0.5
    aggregated = aggregate_weights([arr1, arr2], [1.0, 1.0])
    np.testing.assert_allclose(aggregated[0], np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(aggregated[1], np.array([[20.0, 30.0]]))

    # Unequal weights 0.25, 0.75
    aggregated_unequal = aggregate_weights([arr1, arr2], [1.0, 3.0])
    np.testing.assert_allclose(aggregated_unequal[0], 0.25 * arr1[0] + 0.75 * arr2[0])


def test_client_fit_and_parameters():
    """Verify HospitalFlowerClient can set/get parameters and perform local fit."""
    partitions = create_hospital_partitions(random_seed=42)
    h0 = partitions[0]
    client = HospitalFlowerClient(h0, batch_size=16, lr=0.01)

    initial_params = client.get_parameters(config={})
    assert len(initial_params) > 0

    updated_params, n_samples, fit_res = client.fit(
        parameters=initial_params,
        config={"local_epochs": 1, "mu": 0.0, "lr": 0.01},
    )

    assert n_samples == h0.n_train
    assert "train_loss" in fit_res
    assert fit_res["train_loss"] > 0.0
    assert len(updated_params) == len(initial_params)


def test_fedprox_proximal_regularization():
    """Verify FedProx proximal term penalizes model parameter divergence."""
    partitions = create_hospital_partitions(random_seed=42)
    h0 = partitions[0]

    # Client without proximal term (mu = 0)
    client_no_prox = HospitalFlowerClient(h0, batch_size=32, lr=0.01, mu=0.0)
    # Client with strong proximal term (mu = 10.0)
    client_prox = HospitalFlowerClient(h0, batch_size=32, lr=0.01, mu=10.0)

    initial_params = client_no_prox.get_parameters(config={})

    # Fit with mu=0 vs mu=10.0
    params_no_prox, _, _ = client_no_prox.fit(initial_params, config={"local_epochs": 2, "mu": 0.0})
    params_prox, _, _ = client_prox.fit(initial_params, config={"local_epochs": 2, "mu": 10.0})

    # Drift from initial weights should be strictly smaller with proximal constraint
    drift_no_prox = sum(np.linalg.norm(p - init) for p, init in zip(params_no_prox, initial_params))
    drift_prox = sum(np.linalg.norm(p - init) for p, init in zip(params_prox, initial_params))

    assert drift_prox < drift_no_prox, f"Expected drift_prox ({drift_prox}) < drift_no_prox ({drift_no_prox})"
