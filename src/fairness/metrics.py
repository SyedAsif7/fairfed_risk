"""
FairFed Risk - Fairness Metrics Engine
Computes Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
across biological sex, age tiers, and hospital regions using Fairlearn and NumPy.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import roc_auc_score

TARGET_NAMES = ["diabetes", "heart_disease", "ckd"]


def compute_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: np.ndarray,
    sensitive: np.ndarray,  # columns: [sex, age_group, hospital_id]
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive fairness diagnostics across all protected demographic attributes:
      1. Sex (Female: 0, Male: 1)
      2. Age Tier (<45: 0, 45-60: 1, >60: 2)
      3. Hospital Region (0 to 5)
    """
    audit: Dict[str, Any] = {
        "by_sex": _audit_single_attribute(y_true, y_pred, masks, sensitive[:, 0], "sex", threshold),
        "by_age": _audit_single_attribute(y_true, y_pred, masks, sensitive[:, 1], "age_group", threshold),
        "by_region": _audit_single_attribute(y_true, y_pred, masks, sensitive[:, 2], "hospital_id", threshold),
    }

    # Summary overall macro fairness
    audit["overall"] = {
        "sex_dpd": audit["by_sex"]["macro_dpd"],
        "sex_eod": audit["by_sex"]["macro_eod"],
        "age_dpd": audit["by_age"]["macro_dpd"],
        "age_eod": audit["by_age"]["macro_eod"],
        "region_dpd": audit["by_region"]["macro_dpd"],
        "region_eod": audit["by_region"]["macro_eod"],
    }
    return audit


def _audit_single_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: np.ndarray,
    group_col: np.ndarray,
    attr_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Audits fairness across tasks for a single sensitive attribute."""
    unique_groups = np.unique(group_col)
    res: Dict[str, Any] = {
        "attribute": attr_name,
        "groups": [int(g) for g in unique_groups],
        "per_disease": {},
    }

    dpd_list = []
    eod_list = []

    for k, disease in enumerate(TARGET_NAMES):
        obs = masks[:, k] == 1
        n_obs = int(obs.sum())

        if n_obs < 10:
            res["per_disease"][disease] = {
                "dpd": float("nan"),
                "eod": float("nan"),
                "worst_group_auroc": float("nan"),
            }
            continue

        t_k = y_true[obs, k].astype(int)
        p_k = y_pred[obs, k]
        b_k = (p_k >= threshold).astype(int)
        g_k = group_col[obs]

        # Use Fairlearn metrics with try/except guards for single-class subgroups
        try:
            dpd = float(demographic_parity_difference(t_k, b_k, sensitive_features=g_k))
        except Exception:
            # Fallback direct computation
            sr_vals = [float(b_k[g_k == g].mean()) for g in unique_groups if (g_k == g).sum() > 0]
            dpd = float(np.max(sr_vals) - np.min(sr_vals)) if len(sr_vals) >= 2 else 0.0

        try:
            eod = float(equalized_odds_difference(t_k, b_k, sensitive_features=g_k))
        except Exception:
            # Fallback manual calculation
            tpr_vals, fpr_vals = [], []
            for g in unique_groups:
                pos = (g_k == g) & (t_k == 1)
                neg = (g_k == g) & (t_k == 0)
                if pos.sum() > 0:
                    tpr_vals.append(float(b_k[pos].mean()))
                if neg.sum() > 0:
                    fpr_vals.append(float(b_k[neg].mean()))
            tpr_diff = float(np.max(tpr_vals) - np.min(tpr_vals)) if len(tpr_vals) >= 2 else 0.0
            fpr_diff = float(np.max(fpr_vals) - np.min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0
            eod = max(tpr_diff, fpr_diff)

        # Subgroup AUROCs
        group_aucs = {}
        for g in unique_groups:
            idx_g = g_k == g
            if idx_g.sum() >= 5 and len(np.unique(t_k[idx_g])) > 1:
                try:
                    group_aucs[int(g)] = float(roc_auc_score(t_k[idx_g], p_k[idx_g]))
                except Exception:
                    group_aucs[int(g)] = float("nan")
            else:
                group_aucs[int(g)] = float("nan")

        valid_aucs = [v for v in group_aucs.values() if not np.isnan(v)]
        worst_auc = float(np.min(valid_aucs)) if valid_aucs else float("nan")

        res["per_disease"][disease] = {
            "dpd": dpd,
            "eod": eod,
            "worst_group_auroc": worst_auc,
            "subgroup_aurocs": group_aucs,
        }
        dpd_list.append(dpd)
        eod_list.append(eod)

    res["macro_dpd"] = float(np.nanmean(dpd_list)) if dpd_list else 0.0
    res["macro_eod"] = float(np.nanmean(eod_list)) if eod_list else 0.0
    return res
