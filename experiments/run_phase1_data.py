"""
FairFed Risk - Phase 1 Runner: Data Ingestion, Harmonization, and Non-IID Partitioning
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.download import download_all_datasets
from src.data.harmonize import load_harmonized_dataset, FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.partition import create_hospital_partitions, get_centralized_dataset, HOSPITAL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase1(random_seed: int = 42) -> dict:
    """Executes Phase 1 data pipeline and saves artifacts."""
    logger.info("=== Starting Phase 1: Ingestion, Harmonization & Partitioning ===")
    
    # 1. Download
    raw_files = download_all_datasets(raw_dir=Path("data/raw"))
    
    # 2. Harmonize
    df_harmonized = load_harmonized_dataset(raw_dir=Path("data/raw"))
    df_harmonized.to_csv("data/processed/harmonized_clinical_records.csv", index=False)
    
    # 3. Partition
    partitions = create_hospital_partitions(df=df_harmonized, test_size=0.20, random_seed=random_seed)
    
    # 4. Summary metrics
    summary = {
        "total_records": len(df_harmonized),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "n_targets": len(TARGET_COLUMNS),
        "targets": TARGET_COLUMNS,
        "n_hospitals": len(partitions),
        "hospitals": {},
    }
    
    for hid, h_data in partitions.items():
        summary["hospitals"][hid] = {
            "name": HOSPITAL_NAMES[hid],
            "n_train": h_data.n_train,
            "n_test": h_data.n_test,
            "total": h_data.n_train + h_data.n_test,
            "train_observed_masks": {
                t: int(h_data.train_masks[:, i].sum()) for i, t in enumerate(TARGET_COLUMNS)
            },
            "train_positive_rates": {
                t: float(h_data.train_y[:, i][h_data.train_masks[:, i] == 1].mean())
                if h_data.train_masks[:, i].sum() > 0 else 0.0
                for i, t in enumerate(TARGET_COLUMNS)
            },
            "test_observed_masks": {
                t: int(h_data.test_masks[:, i].sum()) for i, t in enumerate(TARGET_COLUMNS)
            },
            "sex_ratio_male_train": float((h_data.train_sensitive[:, 0] == 1).mean()),
            "age_groups_train": {
                "<45": int((h_data.train_sensitive[:, 1] == 0).sum()),
                "45-60": int((h_data.train_sensitive[:, 1] == 1).sum()),
                ">60": int((h_data.train_sensitive[:, 1] == 2).sum()),
            },
        }
        
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "phase1_partition_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Saved Phase 1 partition summary to %s", out_file)
    return summary


if __name__ == "__main__":
    summary = run_phase1()
    print("\nPhase 1 Execution Summary:")
    print(f"Total Cohort Records: {summary['total_records']}")
    print(f"Hospitals Configured: {summary['n_hospitals']}")
    for hid, info in summary["hospitals"].items():
        print(f"  [{hid}] {info['name']}: N={info['total']} (Train={info['n_train']}, Test={info['n_test']}) | Male %: {info['sex_ratio_male_train']*100:.1f}%")
