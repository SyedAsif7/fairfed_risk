"""
FairFed Risk - Phase 4 Runner: Differential Privacy (DP-SGD) with Opacus
Evaluates federated learning across privacy budgets: epsilon in {1.0, 3.0, 8.0, infinity}, delta = 1e-5.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.partition import create_hospital_partitions
from src.federated.server import FederatedSimulationServer
from src.federated.strategies import set_model_weights_from_ndarrays
from src.models.multitask_mlp import MultiTaskRiskMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase4_dp(config_path: str = "configs/fed_dp.yaml") -> Dict[str, Any]:
    """Runs Phase 4 DP-SGD federated experiments across epsilon grid."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    fed_cfg = cfg["federated"]
    priv_cfg = cfg["privacy"]
    seed = fed_cfg.get("random_seed", 42)
    device = torch.device(fed_cfg.get("device", "cpu"))

    logger.info("=== Starting Phase 4: Differential Privacy (DP-SGD Grid) ===")
    partitions = create_hospital_partitions(random_seed=seed)

    results: Dict[str, Any] = {"experiment": "Phase 4 Differential Privacy", "epsilon_runs": {}}
    table_rows = []

    out_dir = Path(cfg.get("output_dir", "results/metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = Path(cfg.get("checkpoints_dir", "results/checkpoints"))
    chk_dir.mkdir(parents=True, exist_ok=True)

    eps_grid = priv_cfg["epsilon_grid"]

    for eps_target in eps_grid:
        eps_float = float(eps_target)
        eps_label = "inf" if np.isinf(eps_float) else f"eps_{eps_float:.1f}"
        logger.info("--- Running Simulation for Epsilon Target: %s ---", eps_label)

        server = FederatedSimulationServer(
            hospital_partitions=partitions,
            strategy_name=fed_cfg.get("strategy", "fedprox"),
            num_rounds=fed_cfg["num_rounds"],
            local_epochs=fed_cfg["local_epochs"],
            batch_size=fed_cfg["batch_size"],
            lr=fed_cfg["learning_rate"],
            mu=float(fed_cfg.get("mu", 0.01)),
            target_epsilon=eps_float,
            max_grad_norm=float(priv_cfg.get("max_grad_norm", 1.0)),
            delta=float(priv_cfg.get("delta", 1e-5)),
            device=device,
            random_seed=seed,
        )

        history = server.run()
        results["epsilon_runs"][eps_label] = history

        # Save model checkpoint
        final_model = MultiTaskRiskMLP(input_dim=7, hidden_dim1=64, hidden_dim2=32).to(device)
        set_model_weights_from_ndarrays(final_model, history["final_parameters"])
        torch.save(final_model.state_dict(), chk_dir / f"fed_dp_{eps_label}.pt")

        final_m = history["final_metrics"]
        actual_eps_spent = history["epsilon_spent"][-1] if history["epsilon_spent"] else float("inf")

        table_rows.append({
            "Target_Epsilon": "inf" if np.isinf(eps_float) else eps_float,
            "Actual_Eps_Spent": "inf" if np.isinf(actual_eps_spent) else round(actual_eps_spent, 2),
            "Delta": float(priv_cfg["delta"]),
            "Noise_Multiplier_Sigma": round(history["noise_multiplier"], 4),
            "Macro_AUROC": round(final_m["macro"]["auroc"], 4),
            "Macro_AUPRC": round(final_m["macro"]["auprc"], 4),
            "Macro_Brier": round(final_m["macro"]["brier"], 4),
            "Diabetes_AUROC": round(final_m["per_disease"]["diabetes"]["auroc"], 4),
            "Heart_AUROC": round(final_m["per_disease"]["heart_disease"]["auroc"], 4),
            "CKD_AUROC": round(final_m["per_disease"]["ckd"]["auroc"], 4),
            "Sex_DPD": round(final_m["fairness"]["by_sex"]["macro_dpd"], 4),
            "Sex_EOD": round(final_m["fairness"]["by_sex"]["macro_eod"], 4),
            "Age_DPD": round(final_m["fairness"]["by_age"]["macro_dpd"], 4),
            "Age_EOD": round(final_m["fairness"]["by_age"]["macro_eod"], 4),
        })

    # Save outputs (clean out large numpy arrays)
    clean_results = {"experiment": results["experiment"], "epsilon_runs": {}}
    for e_key, h_dict in results["epsilon_runs"].items():
        clean_h = {k: v for k, v in h_dict.items() if k != "final_parameters"}
        clean_results["epsilon_runs"][e_key] = clean_h

    with open(out_dir / "phase4_dp.json", "w") as f:
        json.dump(clean_results, f, indent=2)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out_dir / "phase4_dp_table.csv", index=False)
    logger.info("Saved Phase 4 DP results to %s", out_dir / "phase4_dp_table.csv")

    return clean_results


if __name__ == "__main__":
    run_phase4_dp()
