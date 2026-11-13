"""
FairFed Risk - Server-Side Fairness-Aware Aggregation Reweighting
Implements the FairFed dynamic client reweighting strategy (Ezzeldin et al., 2021).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


class FairFedWeightUpdater:
    """
    Dynamically adjusts server aggregation weights based on institutional fairness metrics:
      w_k^{t+1} = max(w_k^t - beta * (F_k^t - avg(F^t)), min_weight)
      normalized to sum to 1.0.
    """

    def __init__(
        self,
        client_ids: List[int],
        initial_sample_weights: Dict[int, float],
        beta: float = 1.0,
        min_weight: float = 0.02,
        momentum: float = 0.8,
    ):
        self.client_ids = client_ids
        self.weights = initial_sample_weights.copy()
        self.beta = float(beta)
        self.min_weight = float(min_weight)
        self.momentum = float(momentum)
        self.weight_history: List[Dict[int, float]] = [self.weights.copy()]

    def update_weights(self, client_fairness_gaps: Dict[int, float]) -> Dict[int, float]:
        """
        Updates client weights given current validation fairness gaps F_k^t.
        """
        if self.beta <= 0.0 or not client_fairness_gaps:
            return self.weights.copy()

        valid_gaps = [g for g in client_fairness_gaps.values() if not np.isnan(g)]
        if not valid_gaps:
            return self.weights.copy()

        avg_gap = float(np.mean(valid_gaps))

        new_weights: Dict[int, float] = {}
        for cid in self.client_ids:
            gap = client_fairness_gaps.get(cid, avg_gap)
            if np.isnan(gap):
                gap = avg_gap
            diff = gap - avg_gap
            target_w = max(self.weights[cid] - self.beta * diff, self.min_weight)
            # Smooth update with momentum
            smoothed_w = self.momentum * self.weights[cid] + (1.0 - self.momentum) * target_w
            new_weights[cid] = max(smoothed_w, self.min_weight)

        # Normalize to probability simplex
        total_w = sum(new_weights.values())
        self.weights = {cid: w / total_w for cid, w in new_weights.items()}
        self.weight_history.append(self.weights.copy())

        return self.weights.copy()
