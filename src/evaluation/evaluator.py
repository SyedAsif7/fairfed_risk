"""
FairFed Risk - Evaluation & Subgroup Metrics Engine
Computes clinical risk prediction performance (AUROC, AUPRC, Brier Calibration)
and subgroup fairness disparities across biological sex and age tiers.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

TARGET_NAMES = ["diabetes", "heart_disease", "ckd"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: np.ndarray,
    sensitive: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates multi-task disease risk predictions.
    Args:
      y_true: (N, 3) true binary disease states
      y_pred: (N, 3) predicted probabilities in [0, 1]
      masks: (N, 3) observation masks {0, 1}
      sensitive: (N, 3) sensitive attributes [sex, age_group, hospital_id]
      threshold: decision threshold for binary classification
    Returns:
      Comprehensive metrics dictionary including per-disease and subgroup metrics.
    """
    metrics: Dict[str, Any] = {
        "per_disease": {},
        "macro": {},
        "fairness": {},
    }

    aurocs: List[float] = []
    auprcs: List[float] = []
    briers: List[float] = []

    for k, disease in enumerate(TARGET_NAMES):
        obs = masks[:, k] == 1
        n_obs = int(obs.sum())
        d_metrics: Dict[str, Any] = {"n_observed": n_obs}

        if n_obs > 5:
            targets_k = y_true[obs, k]
            preds_k = y_pred[obs, k]
            n_pos = int((targets_k == 1).sum())
            n_neg = int((targets_k == 0).sum())
            d_metrics["n_pos"] = n_pos
            d_metrics["n_neg"] = n_neg

            if n_pos > 0 and n_neg > 0:
                try:
                    auc = float(roc_auc_score(targets_k, preds_k))
                    d_metrics["auroc"] = auc
                    aurocs.append(auc)
                except Exception:
                    d_metrics["auroc"] = float("nan")

                try:
                    ap = float(average_precision_score(targets_k, preds_k))
                    d_metrics["auprc"] = ap
                    auprcs.append(ap)
                except Exception:
                    d_metrics["auprc"] = float("nan")

                try:
                    brier = float(brier_score_loss(targets_k, preds_k))
                    d_metrics["brier"] = brier
                    briers.append(brier)
                except Exception:
                    d_metrics["brier"] = float("nan")
            else:
                d_metrics["auroc"] = float("nan")
                d_metrics["auprc"] = float("nan")
                d_metrics["brier"] = float("nan")
        else:
            d_metrics["auroc"] = float("nan")
            d_metrics["auprc"] = float("nan")
            d_metrics["brier"] = float("nan")

        metrics["per_disease"][disease] = d_metrics

    # Macro averages
    metrics["macro"]["auroc"] = float(np.nanmean(aurocs)) if aurocs else float("nan")
    metrics["macro"]["auprc"] = float(np.nanmean(auprcs)) if auprcs else float("nan")
    metrics["macro"]["brier"] = float(np.nanmean(briers)) if briers else float("nan")

    # Subgroup fairness metrics if sensitive attributes are supplied
    if sensitive is not None:
        sex_col = sensitive[:, 0]
        age_col = sensitive[:, 1]
        metrics["fairness"]["by_sex"] = compute_subgroup_fairness(
            y_true, y_pred, masks, group_labels=sex_col, group_name="sex", threshold=threshold
        )
        metrics["fairness"]["by_age"] = compute_subgroup_fairness(
            y_true, y_pred, masks, group_labels=age_col, group_name="age_group", threshold=threshold
        )

    return metrics


def compute_subgroup_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: np.ndarray,
    group_labels: np.ndarray,
    group_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
    across distinct demographic groups for each disease task.
    """
    groups = np.unique(group_labels)
    fairness_res: Dict[str, Any] = {"groups": [int(g) for g in groups], "metrics_per_disease": {}}

    dpds = []
    eods = []

    for k, disease in enumerate(TARGET_NAMES):
        obs = masks[:, k] == 1
        t_k = y_true[:, k]
        p_k = y_pred[:, k]
        binary_pred_k = (p_k >= threshold).astype(int)

        group_tpr = {}
        group_fpr = {}
        group_selection_rate = {}
        group_auroc = {}

        for g in groups:
            mask_g = obs & (group_labels == g)
            n_g = int(mask_g.sum())
            if n_g > 0:
                # Selection rate P(Y_hat = 1 | A = g)
                sel_rate = float(binary_pred_k[mask_g].mean())
                group_selection_rate[int(g)] = sel_rate

                # True positive & false positive rates
                pos_idx = mask_g & (t_k == 1)
                neg_idx = mask_g & (t_k == 0)

                tpr = float(binary_pred_k[pos_idx].mean()) if pos_idx.sum() > 0 else float("nan")
                fpr = float(binary_pred_k[neg_idx].mean()) if neg_idx.sum() > 0 else float("nan")
                group_tpr[int(g)] = tpr
                group_fpr[int(g)] = fpr

                if pos_idx.sum() > 0 and neg_idx.sum() > 0:
                    try:
                        g_auc = float(roc_auc_score(t_k[mask_g], p_k[mask_g]))
                        group_auroc[int(g)] = g_auc
                    except Exception:
                        group_auroc[int(g)] = float("nan")
                else:
                    group_auroc[int(g)] = float("nan")

        # Demographic Parity Difference: max(SR) - min(SR)
        sr_vals = [v for v in group_selection_rate.values() if not np.isnan(v)]
        dpd = float(np.max(sr_vals) - np.min(sr_vals)) if len(sr_vals) >= 2 else 0.0

        # Equalized Odds Difference: max(max(TPR) - min(TPR), max(FPR) - min(FPR))
        tpr_vals = [v for v in group_tpr.values() if not np.isnan(v)]
        fpr_vals = [v for v in group_fpr.values() if not np.isnan(v)]
        tpr_diff = float(np.max(tpr_vals) - np.min(tpr_vals)) if len(tpr_vals) >= 2 else 0.0
        fpr_diff = float(np.max(fpr_vals) - np.min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0
        eod = max(tpr_diff, fpr_diff)

        # Worst-group AUROC
        auc_vals = [v for v in group_auroc.values() if not np.isnan(v)]
        worst_auc = float(np.min(auc_vals)) if auc_vals else float("nan")

        fairness_res["metrics_per_disease"][disease] = {
            "dpd": dpd,
            "eod": eod,
            "worst_group_auroc": worst_auc,
            "group_auroc": group_auroc,
            "group_selection_rate": group_selection_rate,
            "group_tpr": group_tpr,
            "group_fpr": group_fpr,
        }
        dpds.append(dpd)
        eods.append(eod)

    fairness_res["macro_dpd"] = float(np.mean(dpds)) if dpds else 0.0
    fairness_res["macro_eod"] = float(np.mean(eods)) if eods else 0.0
    return fairness_res


def evaluate_torch_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Helper to evaluate a PyTorch model on a DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_sensitive = []

    with torch.no_grad():
        for x, y, mask, sensitive in data_loader:
            x = x.to(device)
            probs = model(x)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_sensitive.append(sensitive.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    masks = np.vstack(all_masks)
    sensitive = np.vstack(all_sensitive)

    return evaluate_predictions(targets, preds, masks, sensitive)
