"""
FairFed Risk - Phase 2 Baselines: Centralized Training vs Local-Only Siloed Training
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

from src.data.partition import create_hospital_partitions, get_centralized_dataset, HOSPITAL_NAMES
from src.data.dataset import ClinicalMultiTaskDataset, build_dataloaders
from src.models.multitask_mlp import MultiTaskRiskMLP, MaskedMultiTaskLoss, compute_pos_weights
from src.evaluation.evaluator import evaluate_torch_model, evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 40,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    pos_weights: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Standard training loop using Adam optimizer and MaskedMultiTaskLoss."""
    model.to(device)
    model.train()
    criterion = MaskedMultiTaskLoss(pos_weights=pos_weights.to(device) if pos_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, mask, _ in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(x)
            loss = criterion(logits, y, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

    return loss_history


def run_phase2_baselines(config_path: str = "configs/baselines.yaml") -> Dict[str, Any]:
    """Runs centralized and local baseline experiments."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["dataset"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg["training"].get("device", "cpu"))

    logger.info("=== Starting Phase 2: Baselines (Centralized vs Local-Only) ===")
    partitions = create_hospital_partitions(random_seed=seed)

    # 1. Centralized Dataset
    tr_x, tr_y, tr_m, tr_s, te_x, te_y, te_m, te_s = get_centralized_dataset(partitions)
    cent_train_ds = ClinicalMultiTaskDataset(tr_x, tr_y, tr_m, tr_s)
    cent_test_ds = ClinicalMultiTaskDataset(te_x, te_y, te_m, te_s)

    cent_train_loader = torch.utils.data.DataLoader(
        cent_train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    cent_test_loader = torch.utils.data.DataLoader(
        cent_test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    # Compute task positive weights
    pos_weights_cent = compute_pos_weights(
        torch.as_tensor(tr_y, dtype=torch.float32),
        torch.as_tensor(tr_m, dtype=torch.float32),
    )

    # Train Centralized Model
    logger.info("Training Centralized Pooled Model...")
    cent_model = MultiTaskRiskMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim1=cfg["model"]["hidden_dim1"],
        hidden_dim2=cfg["model"]["hidden_dim2"],
    )
    cent_losses = train_model(
        cent_model,
        cent_train_loader,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        pos_weights=pos_weights_cent,
        device=device,
    )

    cent_global_metrics = evaluate_torch_model(cent_model, cent_test_loader, device=device)
    logger.info(
        "Centralized Global Test -> Macro AUROC: %.4f | Macro AUPRC: %.4f | Brier: %.4f",
        cent_global_metrics["macro"]["auroc"],
        cent_global_metrics["macro"]["auprc"],
        cent_global_metrics["macro"]["brier"],
    )

    # Evaluate centralized model per hospital test partition
    cent_per_hosp_metrics = {}
    for hid, h_data in partitions.items():
        _, h_test_loader = build_dataloaders(h_data, batch_size=cfg["training"]["batch_size"])
        h_metrics = evaluate_torch_model(cent_model, h_test_loader, device=device)
        cent_per_hosp_metrics[hid] = h_metrics

    # 2. Local-Only Models per Hospital
    local_results = {}
    for hid, h_data in partitions.items():
        logger.info("Training Local-Only Model for Hospital %d (%s)...", hid, HOSPITAL_NAMES[hid])
        h_tr_loader, h_te_loader = build_dataloaders(h_data, batch_size=cfg["training"]["batch_size"])

        h_pos_w = compute_pos_weights(
            torch.as_tensor(h_data.train_y, dtype=torch.float32),
            torch.as_tensor(h_data.train_masks, dtype=torch.float32),
        )

        loc_model = MultiTaskRiskMLP(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim1=cfg["model"]["hidden_dim1"],
            hidden_dim2=cfg["model"]["hidden_dim2"],
        )
        train_model(
            loc_model,
            h_tr_loader,
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["learning_rate"],
            pos_weights=h_pos_w,
            device=device,
        )

        # Evaluated on own local test set
        local_test_eval = evaluate_torch_model(loc_model, h_te_loader, device=device)
        # Evaluated on global pooled test set (generalization check)
        global_test_eval = evaluate_torch_model(loc_model, cent_test_loader, device=device)

        local_results[hid] = {
            "name": HOSPITAL_NAMES[hid],
            "local_test_metrics": local_test_eval,
            "global_test_metrics": global_test_eval,
        }
        logger.info(
            "Hospital %d -> Local Test AUROC: %.4f | Global Test AUROC: %.4f",
            hid,
            local_test_eval["macro"]["auroc"],
            global_test_eval["macro"]["auprc"],
        )

    # Assemble Baseline Results Report
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = Path("results/checkpoints")
    chk_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cent_model.state_dict(), chk_dir / "centralized_model.pt")

    results_data = {
        "experiment": "Phase 2 Baselines",
        "centralized": {
            "global_test": cent_global_metrics,
            "per_hospital_test": cent_per_hosp_metrics,
            "final_loss": cent_losses[-1],
        },
        "local_only": local_results,
    }

    with open(out_dir / "phase2_baselines.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Build Comparison CSV
    rows = []
    # Centralized row
    rows.append({
        "Model": "Centralized Pooled (Upper Bound)",
        "Trained_On": "All 6 Hospitals (1669 patients)",
        "Tested_On": "Global Held-Out Test (419 patients)",
        "Macro_AUROC": round(cent_global_metrics["macro"]["auroc"], 4),
        "Macro_AUPRC": round(cent_global_metrics["macro"]["auprc"], 4),
        "Macro_Brier": round(cent_global_metrics["macro"]["brier"], 4),
        "Sex_DPD": round(cent_global_metrics["fairness"]["by_sex"]["macro_dpd"], 4),
        "Sex_EOD": round(cent_global_metrics["fairness"]["by_sex"]["macro_eod"], 4),
        "Age_DPD": round(cent_global_metrics["fairness"]["by_age"]["macro_dpd"], 4),
        "Age_EOD": round(cent_global_metrics["fairness"]["by_age"]["macro_eod"], 4),
    })

    # Local rows (evaluated globally)
    for hid, info in local_results.items():
        g_met = info["global_test_metrics"]
        rows.append({
            "Model": f"Local-Only: {HOSPITAL_NAMES[hid]}",
            "Trained_On": f"Hospital {hid} only",
            "Tested_On": "Global Held-Out Test (419 patients)",
            "Macro_AUROC": round(g_met["macro"]["auroc"], 4) if not np.isnan(g_met["macro"]["auroc"]) else None,
            "Macro_AUPRC": round(g_met["macro"]["auprc"], 4) if not np.isnan(g_met["macro"]["auprc"]) else None,
            "Macro_Brier": round(g_met["macro"]["brier"], 4),
            "Sex_DPD": round(g_met["fairness"]["by_sex"]["macro_dpd"], 4),
            "Sex_EOD": round(g_met["fairness"]["by_sex"]["macro_eod"], 4),
            "Age_DPD": round(g_met["fairness"]["by_age"]["macro_dpd"], 4),
            "Age_EOD": round(g_met["fairness"]["by_age"]["macro_eod"], 4),
        })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(out_dir / "phase2_baselines_table.csv", index=False)
    logger.info("Saved baseline metrics to %s and CSV table to %s", out_dir / "phase2_baselines.json", out_dir / "phase2_baselines_table.csv")

    return results_data


if __name__ == "__main__":
    run_phase2_baselines()
