"""
FairFed Risk - Scientific Plotting & Publication Figures
Generates publication-quality figures:
  1. Utility vs. Privacy Budget (Macro AUROC vs Epsilon)
  2. Pareto Frontier: Utility vs. Fairness Gap (AUROC vs DPD)
  3. Demographic Subgroup Disparity under Differential Privacy Noise
  4. Multi-Disease Sensitivity Profiles under Joint Constraints
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_accuracy_vs_epsilon(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 1: Macro AUROC vs Epsilon Budget comparing No-Fairness vs With-Fairness."""
    plt.figure(figsize=(8, 5))
    
    # Map epsilon values to plotting coordinates
    eps_order = ["eps_1.0", "eps_3.0", "eps_8.0", "non_private"]
    eps_labels = [r"$\epsilon=1.0$", r"$\epsilon=3.0$", r"$\epsilon=8.0$", r"$\epsilon=\infty$ (No DP)"]
    x_indices = np.arange(len(eps_order))

    for fair_mode, color, marker, label in [
        ("No Mitigation", "#1f77b4", "o", "Standard FedProx (No Fairness)"),
        ("Fairness Mitigation", "#2ca02c", "s", "FairFed Risk (Fairness Mitigation)"),
    ]:
        sub = summary_df[summary_df["Fairness_Regime"] == fair_mode].copy()
        sub["sort_key"] = sub["Epsilon_Tier"].map(lambda x: eps_order.index(x) if x in eps_order else 99)
        sub = sub.sort_values("sort_key")

        means = sub["Macro_AUROC_mean"].values
        stds = sub["Macro_AUROC_std"].values

        plt.plot(x_indices, means, marker=marker, color=color, linewidth=2.2, markersize=8, label=label)
        plt.fill_between(x_indices, means - stds, means + stds, color=color, alpha=0.18)

    plt.xticks(x_indices, eps_labels)
    plt.xlabel("Privacy Guarantee (Differential Privacy Budget)")
    plt.ylabel("Multi-Disease Macro AUROC")
    plt.title("Clinical Risk Prediction Utility vs. Differential Privacy Budget", pad=12)
    plt.ylim(0.65, 0.85)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Figure 1 to %s", out_path)


def plot_accuracy_vs_fairness_gap(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 2: Pareto Frontier of Predictive Utility (AUROC) vs Fairness Gap (Sex DPD)."""
    plt.figure(figsize=(8, 5.5))

    palette = {
        "eps_1.0": "#d62728",
        "eps_3.0": "#ff7f0e",
        "eps_8.0": "#9467bd",
        "non_private": "#1f77b4",
    }
    eps_display = {
        "eps_1.0": r"$\epsilon=1.0$",
        "eps_3.0": r"$\epsilon=3.0$",
        "eps_8.0": r"$\epsilon=8.0$",
        "non_private": r"$\epsilon=\infty$",
    }

    for idx, row in summary_df.iterrows():
        e_tier = row["Epsilon_Tier"]
        f_reg = row["Fairness_Regime"]
        marker = "s" if f_reg == "Fairness Mitigation" else "o"
        color = palette.get(e_tier, "#333333")
        label = f"{eps_display.get(e_tier, e_tier)} ({'Fair' if f_reg == 'Fairness Mitigation' else 'Standard'})"

        plt.errorbar(
            row["Sex_DPD_mean"],
            row["Macro_AUROC_mean"],
            xerr=row["Sex_DPD_std"],
            yerr=row["Macro_AUROC_std"],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            markersize=9,
            label=label,
        )

    plt.xlabel("Demographic Parity Difference across Biological Sex (Lower is Fairer)")
    plt.ylabel("Multi-Disease Macro AUROC (Higher is Better)")
    plt.title("Pareto Trade-Off: Clinical Predictive Utility vs. Subgroup Parity Gap", pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, bbox_to_anchor=(1.04, 1), loc="upper left", title="Configurations")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Figure 2 to %s", out_path)


def plot_subgroup_performance(raw_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 3: Subgroup AUROC across Female vs Male Patients under DP noise."""
    plt.figure(figsize=(9, 5))

    # Aggregate female vs male AUROC
    grouped = raw_df.groupby(["Epsilon_Tier", "Fairness_Regime"])[["Female_AUROC", "Male_AUROC"]].mean().reset_index()

    eps_order = ["eps_1.0", "eps_3.0", "eps_8.0", "non_private"]
    eps_labels = [r"$\epsilon=1.0$", r"$\epsilon=3.0$", r"$\epsilon=8.0$", r"$\epsilon=\infty$"]
    x = np.arange(len(eps_order))
    width = 0.2

    # Standard FedProx
    std_sub = grouped[grouped["Fairness_Regime"] == "No Mitigation"].set_index("Epsilon_Tier").reindex(eps_order)
    fair_sub = grouped[grouped["Fairness_Regime"] == "Fairness Mitigation"].set_index("Epsilon_Tier").reindex(eps_order)

    plt.bar(x - 1.5 * width, std_sub["Female_AUROC"], width, label="Standard - Female", color="#e377c2", alpha=0.85)
    plt.bar(x - 0.5 * width, std_sub["Male_AUROC"], width, label="Standard - Male", color="#17becf", alpha=0.85)
    plt.bar(x + 0.5 * width, fair_sub["Female_AUROC"], width, label="FairFed - Female", color="#d62728", alpha=0.85)
    plt.bar(x + 1.5 * width, fair_sub["Male_AUROC"], width, label="FairFed - Male", color="#2ca02c", alpha=0.85)

    plt.xticks(x, eps_labels)
    plt.xlabel("Privacy Guarantee (Epsilon)")
    plt.ylabel("Subgroup AUROC")
    plt.title("Subgroup Performance Disparities by Biological Sex across Privacy Regimes", pad=12)
    plt.ylim(0.55, 0.90)
    plt.legend(frameon=True, loc="lower right", ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Figure 3 to %s", out_path)


def plot_per_disease_tradeoffs(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 4: Disease-specific resilience (Diabetes vs Heart vs CKD) under DP."""
    plt.figure(figsize=(8.5, 5))
    eps_order = ["eps_1.0", "eps_3.0", "eps_8.0", "non_private"]
    eps_labels = [r"$\epsilon=1.0$", r"$\epsilon=3.0$", r"$\epsilon=8.0$", r"$\epsilon=\infty$"]
    x = np.arange(len(eps_order))

    sub = summary_df[summary_df["Fairness_Regime"] == "Fairness Mitigation"].copy()
    sub["sort_key"] = sub["Epsilon_Tier"].map(lambda k: eps_order.index(k) if k in eps_order else 99)
    sub = sub.sort_values("sort_key")

    plt.plot(x, sub["Diabetes_AUROC_mean"], marker="o", color="#2ca02c", linewidth=2.2, label="Type 2 Diabetes")
    plt.plot(x, sub["CKD_AUROC_mean"], marker="s", color="#1f77b4", linewidth=2.2, label="Chronic Kidney Disease")
    plt.plot(x, sub["Heart_AUROC_mean"], marker="^", color="#d62728", linewidth=2.2, label="Coronary Heart Disease")

    plt.xticks(x, eps_labels)
    plt.xlabel("Privacy Guarantee (Epsilon Budget)")
    plt.ylabel("Disease-Specific AUROC")
    plt.title("Multi-Task Disease AUROC Under Joint Privacy and Fairness Constraints", pad=12)
    plt.ylim(0.50, 0.98)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Figure 4 to %s", out_path)
