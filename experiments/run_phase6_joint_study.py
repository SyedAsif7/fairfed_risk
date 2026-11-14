"""
FairFed Risk - Phase 6 Runner: Core Joint Research Study
Executes full factorial grid (Differential Privacy x Fairness Mitigations) across 5 seeds:
  - 4 Privacy Tiers: epsilon in {1.0, 3.0, 8.0, infinity}
  - 2 Fairness Regimes: No Mitigation vs Fairness Mitigation
  - 5 Random Seeds: 42, 43, 44, 45, 46 (Total 40 simulations)
Logs runs to MLflow, generates summary metrics (Mean +/- Std), and renders publication figures.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import yaml
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.data.partition import create_hospital_partitions
from src.federated.server import FederatedSimulationServer
from src.evaluation.plots import (
    plot_accuracy_vs_epsilon,
    plot_accuracy_vs_fairness_gap,
    plot_subgroup_performance,
    plot_per_disease_tradeoffs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase6_joint_study(config_path: str = "configs/joint_study.yaml") -> Dict[str, Any]:
    """Runs the full 40-simulation joint research grid."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    study_cfg = cfg["study"]
    seeds = study_cfg.get("seeds", [42, 43, 44, 45, 46])
    eps_tiers = cfg["epsilon_tiers"]
    fair_regimes = cfg["fairness_regimes"]
    device = torch.device(study_cfg.get("device", "cpu"))

    out_dir = Path(cfg.get("output_dir", "results/metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow tracking
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "FairFed_Risk_Joint_Study"))

    logger.info("=== Starting Phase 6: Core Joint Study (40 Experimental Runs) ===")
    total_runs = len(seeds) * len(eps_tiers) * len(fair_regimes)
    current_run = 0

    raw_records: List[Dict[str, Any]] = []

    for seed in seeds:
        # Create non-IID partitions with seed
        partitions = create_hospital_partitions(random_seed=seed)

        for e_item in eps_tiers:
            e_tier = e_item["tier"]
            eps_val = float(e_item["epsilon"])

            for f_item in fair_regimes:
                f_name = f_item["name"]
                l_fair = float(f_item["lambda_fair"])
                beta_val = float(f_item["fairfed_beta"])
                strat_name = "fairfed" if beta_val > 0.0 else "fedprox"

                current_run += 1
                logger.info(
                    "[%02d/%02d] Seed=%d | Epsilon=%s | Fairness=%s (lambda=%.2f, beta=%.2f)",
                    current_run,
                    total_runs,
                    seed,
                    e_tier,
                    f_name,
                    l_fair,
                    beta_val,
                )

                with mlflow.start_run(run_name=f"{e_tier}_{f_name.replace(' ', '_')}_s{seed}"):
                    mlflow.log_params({
                        "seed": seed,
                        "epsilon_tier": e_tier,
                        "target_epsilon": eps_val,
                        "fairness_regime": f_name,
                        "lambda_fair": l_fair,
                        "fairfed_beta": beta_val,
                        "strategy": strat_name,
                        "num_rounds": study_cfg["num_rounds"],
                        "mu": study_cfg["mu"],
                    })

                    server = FederatedSimulationServer(
                        hospital_partitions=partitions,
                        strategy_name=strat_name,
                        num_rounds=study_cfg["num_rounds"],
                        local_epochs=study_cfg["local_epochs"],
                        batch_size=study_cfg["batch_size"],
                        lr=study_cfg["learning_rate"],
                        mu=float(study_cfg.get("mu", 0.01)),
                        fairfed_beta=beta_val,
                        fairness_penalty_lambda=l_fair,
                        target_epsilon=eps_val,
                        max_grad_norm=float(study_cfg.get("max_grad_norm", 1.0)),
                        delta=float(study_cfg.get("delta", 1e-5)),
                        device=device,
                        random_seed=seed,
                    )

                    history = server.run()
                    final_m = history["final_metrics"]
                    final_eps = history["epsilon_spent"][-1] if history["epsilon_spent"] else float("inf")

                    # Extract subgroup AUROCs
                    sex_audit = final_m["fairness"]["by_sex"]
                    # Group 0 is female, Group 1 is male
                    female_aucs = [
                        sex_audit["metrics_per_disease"][d]["group_auroc"].get(0, float("nan"))
                        for d in ["diabetes", "heart_disease", "ckd"]
                    ]
                    male_aucs = [
                        sex_audit["metrics_per_disease"][d]["group_auroc"].get(1, float("nan"))
                        for d in ["diabetes", "heart_disease", "ckd"]
                    ]
                    female_macro_auc = float(np.nanmean(female_aucs)) if any(not np.isnan(v) for v in female_aucs) else float("nan")
                    male_macro_auc = float(np.nanmean(male_aucs)) if any(not np.isnan(v) for v in male_aucs) else float("nan")

                    record = {
                        "Seed": seed,
                        "Epsilon_Tier": e_tier,
                        "Target_Epsilon": eps_val if eps_val < float("inf") else "inf",
                        "Actual_Eps_Spent": final_eps if final_eps < float("inf") else "inf",
                        "Fairness_Regime": f_name,
                        "Macro_AUROC": float(final_m["macro"]["auroc"]),
                        "Macro_AUPRC": float(final_m["macro"]["auprc"]),
                        "Macro_Brier": float(final_m["macro"]["brier"]),
                        "Diabetes_AUROC": float(final_m["per_disease"]["diabetes"]["auroc"]),
                        "Heart_AUROC": float(final_m["per_disease"]["heart_disease"]["auroc"]),
                        "CKD_AUROC": float(final_m["per_disease"]["ckd"]["auroc"]),
                        "Sex_DPD": float(final_m["fairness"]["by_sex"]["macro_dpd"]),
                        "Sex_EOD": float(final_m["fairness"]["by_sex"]["macro_eod"]),
                        "Age_DPD": float(final_m["fairness"]["by_age"]["macro_dpd"]),
                        "Age_EOD": float(final_m["fairness"]["by_age"]["macro_eod"]),
                        "Female_AUROC": female_macro_auc,
                        "Male_AUROC": male_macro_auc,
                    }
                    raw_records.append(record)

                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        "macro_auroc": record["Macro_AUROC"],
                        "macro_auprc": record["Macro_AUPRC"],
                        "macro_brier": record["Macro_Brier"],
                        "sex_dpd": record["Sex_DPD"],
                        "sex_eod": record["Sex_EOD"],
                        "age_dpd": record["Age_DPD"],
                        "age_eod": record["Age_EOD"],
                        "female_auroc": record["Female_AUROC"],
                        "male_auroc": record["Male_AUROC"],
                    })

    # Save raw records across all 40 runs
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(out_dir / "phase6_joint_study_raw.csv", index=False)

    # Compute Statistical Aggregates (Mean +/- Std across 5 seeds)
    metric_cols = [
        "Macro_AUROC",
        "Macro_AUPRC",
        "Macro_Brier",
        "Diabetes_AUROC",
        "Heart_AUROC",
        "CKD_AUROC",
        "Sex_DPD",
        "Sex_EOD",
        "Age_DPD",
        "Age_EOD",
        "Female_AUROC",
        "Male_AUROC",
    ]

    summary_rows = []
    grouped = raw_df.groupby(["Epsilon_Tier", "Fairness_Regime"])

    for (e_tier, f_reg), group in grouped:
        row_dict: Dict[str, Any] = {
            "Epsilon_Tier": e_tier,
            "Fairness_Regime": f_reg,
            "N_Seeds": len(group),
        }
        for col in metric_cols:
            vals = group[col].dropna()
            mean_val = float(vals.mean()) if len(vals) > 0 else float("nan")
            std_val = float(vals.std()) if len(vals) > 1 else 0.0
            row_dict[f"{col}_mean"] = round(mean_val, 4)
            row_dict[f"{col}_std"] = round(std_val, 4)
            row_dict[f"{col}_formatted"] = f"{mean_val:.4f} +/- {std_val:.4f}"
        summary_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "phase6_joint_study_summary.csv", index=False)

    # Format Publication Table
    pub_table = []
    for _, r in summary_df.iterrows():
        pub_table.append({
            "Epsilon_Budget": r["Epsilon_Tier"],
            "Fairness_Regime": r["Fairness_Regime"],
            "Macro_AUROC": r["Macro_AUROC_formatted"],
            "Macro_AUPRC": r["Macro_AUPRC_formatted"],
            "Macro_Brier": r["Macro_Brier_formatted"],
            "Sex_DPD": r["Sex_DPD_formatted"],
            "Sex_EOD": r["Sex_EOD_formatted"],
            "Age_DPD": r["Age_DPD_formatted"],
            "Age_EOD": r["Age_EOD_formatted"],
        })
    df_pub = pd.DataFrame(pub_table)
    df_pub.to_csv(out_dir / "phase6_joint_study_table.csv", index=False)

    # Generate Publication Figures
    logger.info("Generating publication-quality figures in %s...", plots_dir)
    plot_accuracy_vs_epsilon(summary_df, plots_dir / "fig1_accuracy_vs_epsilon.png")
    plot_accuracy_vs_fairness_gap(summary_df, plots_dir / "fig2_accuracy_vs_fairness_gap.png")
    plot_subgroup_performance(raw_df, plots_dir / "fig3_subgroup_disparity_under_dp.png")
    plot_per_disease_tradeoffs(summary_df, plots_dir / "fig4_per_disease_tradeoffs.png")

    logger.info("Phase 6 Joint Study complete. Real empirical results logged.")
    return {"raw_df": raw_df, "summary_df": summary_df}


if __name__ == "__main__":
    run_phase6_joint_study()
