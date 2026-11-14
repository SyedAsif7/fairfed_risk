"""
FairFed Risk - Privacy Accountant
Tracks privacy budget spending (epsilon, delta) across federated communication rounds
using Rényi Differential Privacy (RDP).
"""

from __future__ import annotations
import functools
from typing import Optional, Dict, Any
import numpy as np
from opacus.accountants import RDPAccountant


@functools.lru_cache(maxsize=128)
def _solve_noise_multiplier(
    target_epsilon: float,
    target_delta: float = 1e-5,
    sample_rate: float = 0.128,
    total_steps: int = 240,
) -> float:
    """Cached solver for DP-SGD noise multiplier."""
    if target_epsilon == float("inf") or target_epsilon is None or target_epsilon <= 0:
        return 0.0

    low, high = 0.1, 30.0
    for _ in range(35):
        mid = (low + high) / 2.0
        acc = RDPAccountant()
        for _ in range(total_steps):
            acc.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = acc.get_epsilon(delta=target_delta)
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return float(mid)


class PrivacyAccountant:
    """
    RDP-based Privacy Accountant for tracking cumulative (epsilon, delta) spending.
    Computes exact epsilon bounds given Gaussian noise multiplier sigma, subsampling rate q,
    and number of local DP-SGD steps.
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.accountant = RDPAccountant()
        self.step_history: list[Dict[str, float]] = []

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Records a single DP-SGD gradient step."""
        if noise_multiplier > 0:
            self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            self.step_history.append({"noise_multiplier": noise_multiplier, "sample_rate": sample_rate})

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Returns the current cumulative epsilon spent."""
        if not self.step_history:
            return 0.0
        target_delta = delta if delta is not None else self.delta
        try:
            eps = self.accountant.get_epsilon(delta=target_delta)
            return float(eps)
        except Exception:
            return float("nan")

    @classmethod
    def calibrate_noise_multiplier(
        cls,
        target_epsilon: float,
        target_delta: float = 1e-5,
        sample_rate: float = 0.128,
        total_steps: int = 240,
    ) -> float:
        """
        Numerically solves for the noise multiplier sigma required to exhaust exactly
        target_epsilon at target_delta after total_steps.
        Returns 0.0 if target_epsilon is infinity (non-private).
        """
        return _solve_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            sample_rate=float(round(sample_rate, 5)),
            total_steps=int(total_steps),
        )
