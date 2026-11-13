"""
Unit tests for Fairness Metrics, Loss Penalties, and FairFed Dynamic Client Reweighting.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from src.fairness.metrics import compute_fairness_audit
from src.fairness.losses import FairnessRegularizedLoss
from src.fairness.reweighting import FairFedWeightUpdater
from src.models.multitask_mlp import MultiTaskRiskMLP


def test_fairness_audit_computation():
    """Verify compute_fairness_audit calculates non-negative DPD and EOD."""
    np.random.seed(42)
    N = 100
    y_true = np.random.randint(0, 2, (N, 3))
    y_pred = np.random.uniform(0.1, 0.9, (N, 3))
    masks = np.ones((N, 3))
    sensitive = np.column_stack([
        np.random.choice([0, 1], size=N),
        np.random.choice([0, 1, 2], size=N),
        np.random.choice([0, 1, 2, 3, 4, 5], size=N),
    ])

    audit = compute_fairness_audit(y_true, y_pred, masks, sensitive)
    assert "by_sex" in audit
    assert "by_age" in audit
    assert "by_region" in audit
    assert "overall" in audit

    assert 0.0 <= audit["overall"]["sex_dpd"] <= 1.0
    assert 0.0 <= audit["overall"]["age_dpd"] <= 1.0


def test_fairness_loss_penalty_gradients():
    """Verify fairness penalty increases loss and propagates gradients."""
    model = MultiTaskRiskMLP(input_dim=7)
    crit_base = FairnessRegularizedLoss(lambda_fair=0.0)
    crit_fair = FairnessRegularizedLoss(lambda_fair=1.0)

    x = torch.randn(20, 7)
    targets = torch.randint(0, 2, (20, 3)).float()
    masks = torch.ones(20, 3)
    sensitive_sex = torch.tensor([0] * 10 + [1] * 10)

    logits = model.forward_logits(x)
    loss_base = crit_base(logits, targets, masks, sensitive_sex)
    loss_fair = crit_fair(logits, targets, masks, sensitive_sex)

    assert loss_fair.item() >= loss_base.item()

    loss_fair.backward()
    assert model.fc1.weight.grad is not None
    assert torch.abs(model.fc1.weight.grad).sum() > 0


def test_fairfed_weight_reweighting():
    """Verify FairFed updater penalizes unfair clients and normalizes weights."""
    client_ids = [0, 1, 2]
    initial_w = {0: 1/3, 1: 1/3, 2: 1/3}

    updater = FairFedWeightUpdater(client_ids, initial_w, beta=1.5, momentum=0.5)

    # Client 0 is very unfair (gap=0.5), Client 1 is fair (gap=0.1), Client 2 is medium (gap=0.3)
    gaps = {0: 0.5, 1: 0.1, 2: 0.3}
    new_w = updater.update_weights(gaps)

    # Sum must be 1.0
    assert abs(sum(new_w.values()) - 1.0) < 1e-6
    # Fair client 1 should gain weight relative to unfair client 0
    assert new_w[1] > new_w[0], f"Client 1 weight ({new_w[1]}) should exceed Client 0 ({new_w[0]})"
