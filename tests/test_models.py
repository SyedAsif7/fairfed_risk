"""
Unit tests for MultiTaskRiskMLP, MaskedMultiTaskLoss, and Evaluator.
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from src.models.multitask_mlp import MultiTaskRiskMLP, MaskedMultiTaskLoss, compute_pos_weights
from src.evaluation.evaluator import evaluate_predictions


def test_model_forward_dimensions():
    """Verify forward feature extraction, logits, and probability outputs."""
    model = MultiTaskRiskMLP(input_dim=7, hidden_dim1=64, hidden_dim2=32)
    batch_x = torch.randn(10, 7)

    features = model.forward_features(batch_x)
    assert features.shape == (10, 32)

    logits = model.forward_logits(batch_x)
    assert logits.shape == (10, 3)

    probs = model(batch_x)
    assert probs.shape == (10, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_masked_loss_ignores_unobserved():
    """Verify that tasks with mask=0 do not contribute to gradients."""
    model = MultiTaskRiskMLP(input_dim=7)
    criterion = MaskedMultiTaskLoss()

    batch_x = torch.randn(8, 7)
    targets = torch.tensor([[1.0, 0.0, 0.0]] * 8)
    # Mask task 1 and 2 out completely
    masks = torch.tensor([[1.0, 0.0, 0.0]] * 8)

    logits = model.forward_logits(batch_x)
    loss = criterion(logits, targets, masks)
    loss.backward()

    # Head 1 (diabetes) should have gradients
    assert model.head_diabetes.weight.grad is not None
    assert torch.abs(model.head_diabetes.weight.grad).sum() > 0

    # Head 2 and Head 3 (unobserved) should have zero gradients
    assert model.head_heart.weight.grad is None or torch.abs(model.head_heart.weight.grad).sum() == 0
    assert model.head_ckd.weight.grad is None or torch.abs(model.head_ckd.weight.grad).sum() == 0


def test_evaluator_metrics_computation():
    """Verify AUROC, AUPRC, Brier, and Subgroup fairness metrics."""
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
    ])
    y_pred = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.8, 0.2],
        [0.85, 0.7, 0.95],
        [0.1, 0.2, 0.05],
        [0.75, 0.3, 0.1],
        [0.3, 0.85, 0.9],
    ])
    masks = np.ones_like(y_true)
    sensitive = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [0, 2, 1],
        [1, 0, 1],
        [0, 1, 2],
        [1, 2, 2],
    ])

    results = evaluate_predictions(y_true, y_pred, masks, sensitive)
    assert "macro" in results
    assert "auroc" in results["macro"]
    assert results["macro"]["auroc"] > 0.85
    assert "fairness" in results
    assert "by_sex" in results["fairness"]
    assert "macro_dpd" in results["fairness"]["by_sex"]
