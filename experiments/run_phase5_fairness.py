"""
FairFed Risk - Phase 5 Runner: Fairness Auditing & Dual Mitigations
Evaluates:
  1. Standard FedProx (No Fairness Mitigation)
  2. Client-Side Loss Penalty (In-Processing)
  3. Server-Side FairFed Reweighting (Aggregation-Level)
  4. Dual Mitigation (Combined Loss + FairFed Reweighting)
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
from src.fairness.metrics import compute_fairness_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase5_fairness(config_path: str = "configs/fairness.yaml") -> Dict[str, Any]:
    """Runs Phase 5 Fairness experiments."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    fed_cfg = cfg["federated"]
    seed = fed_cfg.get("random_seed", 42)
    device = torch.device(fed_cfg.get("device", "cpu"))

    logger.info("=== Starting Phase 5: Fairness Auditing & Mitigations ===")
    partitions = create_hospital_partitions(random_seed=seed)

    results: Dict[str, Any] = {"experiment": "Phase 5 Fairness Mitigations", "runs": {}}
    table_rows = []

    out_dir = Path(cfg.get("output_dir", "results/metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = Path(cfg.get("checkpoints_dir", "results/checkpoints"))
    chk_dir.mkdir(parents=True, exist_ok=True)

    for mit_cfg in cfg["mitigations"]:
        mit_name = mit_cfg["name"]
        l_fair = float(mit_cfg["lambda_fair"])
        beta_val = float(mit_cfg["fairfed_beta"])
        strat_key = "fairfed" if beta_val > 0.0 else "fedprox"

        logger.info("--- Running Mitigation: %s (lambda=%.2f, beta=%.2f) ---", mit_name, l_fair, beta_val)

        server = FederatedSimulationServer(
            hospital_partitions=partitions,
            strategy_name=strat_key,
            num_rounds=fed_cfg["num_rounds"],
            local_epochs=fed_cfg["local_epochs"],
            batch_size=fed_cfg["batch_size"],
            lr=fed_cfg["learning_rate"],
            mu=float(fed_cfg.get("mu", 0.01)),
            fairfed_beta=beta_val,
            fairness_penalty_lambda=l_fair,
            device=device,
            random_seed=seed,
        )

        history = server.run()
        results["runs"][mit_name] = history

        # Save model checkpoint
        file_tag = mit_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        final_model = MultiTaskRiskMLP(input_dim=7, hidden_dim1=64, hidden_dim2=32).to(device)
        set_model_weights_from_ndarrays(final_model, history["final_parameters"])
        torch.save(final_model.state_dict(), chk_dir / f"fairness_{file_tag}.pt")

        final_m = history["final_metrics"]
        table_rows.append({
            "Mitigation_Strategy": mit_name,
            "Lambda_Fair": l_fair,
            "FairFed_Beta": beta_val,
            "Macro_AUROC": round(final_m["macro"]["auroc"], 4),
            "Macro_AUPRC": round(final_m["macro"]["auprc"], 4),
            "Macro_Brier": round(final_m["macro"]["brier"], 4),
            "Sex_DPD": round(final_m["fairness"]["by_sex"]["macro_dpd"], 4),
            "Sex_EOD": round(final_m["fairness"]["by_sex"]["macro_eod"], 4),
            "Age_DPD": round(final_m["fairness"]["by_age"]["macro_dpd"], 4),
            "Age_EOD": round(final_m["fairness"]["by_age"]["macro_eod"], 4),
        })

    # Save outputs
    clean_results = {"experiment": results["experiment"], "runs": {}}
    for m_key, h_dict in results["runs"].items():
        clean_h = {k: v for k, v in h_dict.items() if k != "final_parameters"}
        clean_results["runs"][m_key] = clean_h

    with open(out_dir / "phase5_fairness.json", "w") as f:
        json.dump(clean_results, f, indent=2)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out_dir / "phase5_fairness_table.csv", index=False)
    logger.info("Saved Phase 5 Fairness results to %s", out_dir / "phase5_fairness_table.csv")

    return clean_results


if __name__ == "__main__":
    run_phase5_fairness()
