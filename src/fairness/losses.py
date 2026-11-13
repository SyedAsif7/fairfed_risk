"""
FairFed Risk - In-Processing Fairness Loss Regularizer
Penalizes disparity between sensitive subgroups during local client backpropagation.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from src.models.multitask_mlp import MaskedMultiTaskLoss


class FairnessRegularizedLoss(nn.Module):
    """
    In-Processing Client Loss that incorporates a demographic parity penalty:
      L_total = L_task + lambda_fair * L_fairness
    where L_fairness penalizes the difference in average predicted risk between
    sensitive demographic subgroups (e.g. Female vs. Male, or Young vs. Old).
    """

    def __init__(
        self,
        pos_weights: Optional[torch.Tensor] = None,
        lambda_fair: float = 0.5,
    ):
        super().__init__()
        self.task_loss_fn = MaskedMultiTaskLoss(pos_weights=pos_weights)
        self.lambda_fair = float(lambda_fair)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        sensitive_attribute: torch.Tensor,  # 1D tensor (e.g. sex: 0 or 1)
    ) -> torch.Tensor:
        """
        Calculates combined multi-task and fairness-regularized loss.
        """
        task_loss = self.task_loss_fn(logits, targets, masks)

        if self.lambda_fair <= 0.0 or sensitive_attribute is None:
            return task_loss

        probs = torch.sigmoid(logits)
        s = sensitive_attribute.float()

        fair_penalty = torch.tensor(0.0, device=logits.device)
        n_tasks_penalized = 0

        # Compute demographic parity gap per observed disease head
        for k in range(probs.shape[1]):
            obs = masks[:, k] == 1
            if obs.sum() > 4:
                p_k = probs[obs, k]
                s_k = s[obs]

                mask_g0 = s_k == 0.0
                mask_g1 = s_k == 1.0

                if mask_g0.sum() > 0 and mask_g1.sum() > 0:
                    mean_p_g0 = torch.mean(p_k[mask_g0])
                    mean_p_g1 = torch.mean(p_k[mask_g1])
                    # L1 difference between subgroup risk predictions
                    fair_penalty = fair_penalty + torch.abs(mean_p_g0 - mean_p_g1)
                    n_tasks_penalized += 1

        if n_tasks_penalized > 0:
            avg_fair_penalty = fair_penalty / n_tasks_penalized
            return task_loss + self.lambda_fair * avg_fair_penalty
        else:
            return task_loss
