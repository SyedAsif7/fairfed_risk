"""
FairFed Risk - Flower Federated Client with Differential Privacy (DP-SGD)
Hospital client implementation with masked multi-task loss, proximal regularization (FedProx),
Opacus differential privacy engine, and privacy accountant.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine

from src.models.multitask_mlp import MultiTaskRiskMLP, MaskedMultiTaskLoss, compute_pos_weights
from src.data.partition import HospitalData
from src.data.dataset import build_dataloaders
from src.evaluation.evaluator import evaluate_torch_model


class HospitalFlowerClient(fl.client.NumPyClient):
    """
    Flower client simulating an individual clinical institution (hospital node).
    Supports:
      - Masked multi-task loss over observed modalities
      - Proximal term for FedProx: (mu / 2) * ||theta - theta^t||^2
      - Differential privacy via Opacus DP-SGD (per-sample clipping + Gaussian noise)
      - Dynamic privacy accounting (epsilon, delta)
      - Fairness regularization hooks
    """

    def __init__(
        self,
        hospital_data: HospitalData,
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
        fairness_penalty_lambda: float = 0.0,
        noise_multiplier: float = 0.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hospital_data = hospital_data
        self.hid = hospital_data.hospital_id
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mu = mu
        self.fairness_penalty_lambda = fairness_penalty_lambda
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.delta = float(delta)

        # Initialize local model
        kwargs = model_kwargs or {"input_dim": 7, "hidden_dim1": 64, "hidden_dim2": 32}
        self.raw_model = MultiTaskRiskMLP(**kwargs).to(self.device)
        self.model = self.raw_model

        # Build data loaders
        self.train_loader, self.test_loader = build_dataloaders(
            hospital_data, batch_size=batch_size, shuffle_train=True
        )

        # Compute positive class weights for this hospital's observed targets
        pos_w = compute_pos_weights(
            torch.as_tensor(hospital_data.train_y, dtype=torch.float32),
            torch.as_tensor(hospital_data.train_masks, dtype=torch.float32),
        )
        self.criterion = MaskedMultiTaskLoss(pos_weights=pos_w.to(self.device))

        # Local optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)

        # DP-SGD Privacy Engine setup
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.is_dp = self.noise_multiplier > 0.0
        if self.is_dp:
            self.privacy_engine = PrivacyEngine(secure_mode=False)
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

    def _get_underlying_model(self) -> torch.nn.Module:
        """Returns unwrapped PyTorch module for direct parameter access."""
        if self.is_dp and hasattr(self.model, "_module"):
            return self.model._module
        return self.model

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extracts model weights as cloned NumPy arrays."""
        underlying = self._get_underlying_model()
        return [val.detach().cpu().numpy().copy() for val in underlying.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets model weights from a list of NumPy arrays."""
        underlying = self._get_underlying_model()
        state_dict = OrderedDict()
        for key, param in zip(underlying.state_dict().keys(), parameters):
            state_dict[key] = torch.tensor(param, device=self.device)
        underlying.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Local client training round with optional FedProx and DP-SGD."""
        self.set_parameters(parameters)
        underlying = self._get_underlying_model()
        self.model.train()

        # Capture global weights theta^t for FedProx proximal loss
        global_params = [p.clone().detach() for p in underlying.parameters()]

        # Config overrides
        epochs = config.get("local_epochs", 2)
        mu = config.get("mu", self.mu)

        epoch_losses = []
        for _ in range(epochs):
            batch_losses = []
            for x, y, mask, sensitive in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                logits = underlying.forward_logits(x)
                task_loss = self.criterion(logits, y, mask)

                # FedProx Proximal Regularization
                if mu > 0.0:
                    prox_loss = 0.0
                    for param, g_param in zip(underlying.parameters(), global_params):
                        prox_loss += torch.sum((param - g_param) ** 2)
                    loss = task_loss + (mu / 2.0) * prox_loss
                else:
                    loss = task_loss

                # Fairness Penalty
                if self.fairness_penalty_lambda > 0.0:
                    probs = torch.sigmoid(logits)
                    sex_attr = sensitive[:, 0].to(self.device).float()
                    fair_loss = 0.0
                    for k in range(probs.shape[1]):
                        obs_k = mask[:, k] == 1
                        if obs_k.sum() > 4:
                            p_k = probs[obs_k, k]
                            s_k = sex_attr[obs_k]
                            cov_k = torch.mean((p_k - torch.mean(p_k)) * (s_k - torch.mean(s_k)))
                            fair_loss += torch.abs(cov_k)
                    loss = loss + self.fairness_penalty_lambda * fair_loss

                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(float(np.mean(batch_losses)))

        num_samples = len(self.hospital_data.train_x)
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Current cumulative privacy budget spent (only compute expensive RDP accountant when requested)
        compute_eps = config.get("compute_eps", True)
        if self.is_dp and self.privacy_engine is not None and compute_eps:
            eps_spent = float(self.privacy_engine.get_epsilon(delta=self.delta))
        else:
            eps_spent = float("inf")

        return (
            self.get_parameters(config={}),
            num_samples,
            {
                "train_loss": avg_loss,
                "hospital_id": self.hid,
                "epsilon_spent": eps_spent,
                "delta": self.delta if self.is_dp else 0.0,
            },
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluates model on local held-out test partition."""
        self.set_parameters(parameters)
        underlying = self._get_underlying_model()
        metrics = evaluate_torch_model(underlying, self.test_loader, device=self.device)
        num_samples = len(self.hospital_data.test_x)
        loss = metrics["macro"].get("brier", 0.0)
        return float(loss), num_samples, {
            "auroc": float(metrics["macro"].get("auroc", 0.0)),
            "auprc": float(metrics["macro"].get("auprc", 0.0)),
            "hospital_id": self.hid,
        }
