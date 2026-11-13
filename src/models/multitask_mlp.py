"""
FairFed Risk - Multi-Task Disease Risk Prediction Model
Shared MLP encoder with disease-specific prediction heads and masked multi-task BCE loss.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskRiskMLP(nn.Module):
    """
    Multi-Task Neural Network for clinical risk prediction:
      - Shared encoder: 7 features -> 64 -> 32
      - LayerNorm instead of BatchNorm for compatibility with DP-SGD (Opacus)
      - Three independent classification heads:
          1. Diabetes Risk (Sigmoid probability)
          2. Coronary Heart Disease Risk (Sigmoid probability)
          3. Chronic Kidney Disease Risk (Sigmoid probability)
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout1: float = 0.2,
        dropout2: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Shared representation trunk
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.norm1 = nn.LayerNorm(hidden_dim1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout2)

        # Task-specific logits heads
        self.head_diabetes = nn.Linear(hidden_dim2, 1)
        self.head_heart = nn.Linear(hidden_dim2, 1)
        self.head_ckd = nn.Linear(hidden_dim2, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 32-dimensional shared latent representation."""
        h = self.drop1(self.act1(self.norm1(self.fc1(x))))
        h = self.drop2(self.act2(self.fc2(h)))
        return h

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw un-squashed logits of shape (B, 3)."""
        z = self.forward_features(x)
        logit_diabetes = self.head_diabetes(z)
        logit_heart = self.head_heart(z)
        logit_ckd = self.head_ckd(z)
        logits = torch.cat([logit_diabetes, logit_heart, logit_ckd], dim=-1)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted probabilities of shape (B, 3)."""
        logits = self.forward_logits(x)
        probs = torch.sigmoid(logits)
        return probs


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked Multi-Task Binary Cross-Entropy Loss with positive class weights.
    Calculates BCE strictly for tasks where the clinical observation mask M_k = 1.
    """

    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.clone().detach())
        else:
            self.pos_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          logits: (B, 3) unnormalized network outputs
          targets: (B, 3) binary true disease states {0, 1}
          masks: (B, 3) binary clinical observation masks {0, 1}
        Returns:
          Scalar loss averaged over all observed task instances.
        """
        # Element-wise BCE with logits
        if self.pos_weights is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weights,
                reduction="none",
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )

        # Apply observation masks
        masked_bce = bce * masks
        total_observed = masks.sum()

        if total_observed > 0:
            return masked_bce.sum() / total_observed
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


def compute_pos_weights(train_targets: torch.Tensor, train_masks: torch.Tensor) -> torch.Tensor:
    """Computes N_neg / N_pos for each task to address class imbalance."""
    weights = []
    for k in range(train_targets.shape[1]):
        obs = train_masks[:, k] == 1
        n_pos = (train_targets[:, k][obs] == 1).sum().item()
        n_neg = (train_targets[:, k][obs] == 0).sum().item()
        w = (n_neg / max(n_pos, 1.0)) if n_pos > 0 else 1.0
        # Clip weight to prevent extreme gradient swings on small local batches
        w = float(np.clip(w, 0.5, 5.0))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
