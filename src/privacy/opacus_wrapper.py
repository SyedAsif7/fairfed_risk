"""
FairFed Risk - Opacus Differential Privacy Wrapper
Wraps PyTorch models and optimizers with Opacus PrivacyEngine for DP-SGD.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional, Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.privacy.accountant import PrivacyAccountant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DPClientEngine:
    """
    Manages local client Differential Privacy (DP-SGD).
    Supports:
      - Target epsilon in {1.0, 3.0, 8.0, infinity}
      - Per-sample gradient clipping (max_grad_norm C)
      - Calibrated Gaussian noise addition
      - Tracking cumulative (epsilon, delta) spending per communication round
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        noise_multiplier: float = 0.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        self.raw_model = model
        self.raw_optimizer = optimizer
        self.raw_loader = train_loader
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.delta = float(delta)

        self.is_private = self.noise_multiplier > 0.0
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.model = model
        self.optimizer = optimizer
        self.loader = train_loader

        if self.is_private:
            self.privacy_engine = PrivacyEngine(secure_mode=False)
            self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
                module=self.raw_model,
                optimizer=self.raw_optimizer,
                data_loader=self.raw_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

    def get_current_epsilon(self) -> float:
        """Returns the cumulative epsilon spent so far."""
        if not self.is_private or self.privacy_engine is None:
            return float("inf")
        try:
            return float(self.privacy_engine.get_epsilon(delta=self.delta))
        except Exception:
            return float("nan")

    def unwrap_model(self) -> nn.Module:
        """Returns underlying un-wrapped model for weight extraction."""
        if self.is_private and hasattr(self.model, "_module"):
            return self.model._module
        return self.model
