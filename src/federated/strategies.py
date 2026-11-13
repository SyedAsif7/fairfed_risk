"""
FairFed Risk - Federated Aggregation Strategies
Implements FedAvg, FedProx, and FairFed client weighting mechanisms with full
multi-disease evaluation and subgroup fairness auditing.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from src.models.multitask_mlp import MultiTaskRiskMLP
from src.evaluation.evaluator import evaluate_torch_model


def aggregate_weights(
    client_parameters: List[List[np.ndarray]],
    client_weights: List[float],
) -> List[np.ndarray]:
    """
    Computes weighted average of client model parameters:
      theta = sum_k (w_k * theta_k) / sum_k(w_k)
    """
    total_weight = sum(client_weights)
    if total_weight <= 0:
        raise ValueError("Total client aggregation weight must be positive.")

    normalized_weights = [w / total_weight for w in client_weights]

    aggregated: List[np.ndarray] = []
    # Iterate over each parameter tensor in the model
    num_tensors = len(client_parameters[0])
    for tensor_idx in range(num_tensors):
        tensor_sum = np.zeros_like(client_parameters[0][tensor_idx], dtype=np.float64)
        for client_idx, client_params in enumerate(client_parameters):
            tensor_sum += normalized_weights[client_idx] * client_params[tensor_idx]
        aggregated.append(tensor_sum.astype(client_parameters[0][tensor_idx].dtype))

    return aggregated


def set_model_weights_from_ndarrays(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Loads list of NumPy arrays into a PyTorch model state_dict."""
    state_dict = OrderedDict()
    for key, param in zip(model.state_dict().keys(), parameters):
        state_dict[key] = torch.tensor(param, device=next(model.parameters()).device)
    model.load_state_dict(state_dict, strict=True)


def evaluate_global_model(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
    global_test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Evaluates an aggregated parameter set against the global multi-center test set."""
    set_model_weights_from_ndarrays(model, parameters)
    metrics = evaluate_torch_model(model, global_test_loader, device=device)
    return metrics
