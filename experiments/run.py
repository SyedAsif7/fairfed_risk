"""
FairFed Risk - Unified Experiment Entrypoint
Allows running any experiment via a single command:
  python experiments/run.py --config configs/fed_dp.yaml
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import yaml

from experiments.run_phase1_data import run_phase1
from experiments.run_phase2_baselines import run_phase2_baselines
from experiments.run_phase3_federated import run_phase3_federated
from experiments.run_phase4_dp import run_phase4_dp
from experiments.run_phase5_fairness import run_phase5_fairness
from experiments.run_phase6_joint_study import run_phase6_joint_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="FairFed Risk - Master Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration file (e.g. configs/fed_dp.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "").lower()
    logger.info("Executing experiment '%s' using config: %s", exp_name, config_path)

    if "baseline" in exp_name:
        run_phase2_baselines(str(config_path))
    elif "simulation" in exp_name or "fedavg" in exp_name or "fedprox" in exp_name:
        run_phase3_federated(str(config_path))
    elif "privacy" in exp_name or "dp" in exp_name:
        run_phase4_dp(str(config_path))
    elif "fairness" in exp_name:
        run_phase5_fairness(str(config_path))
    elif "joint" in exp_name or "study" in exp_name:
        run_phase6_joint_study(str(config_path))
    else:
        logger.warning("Unrecognized experiment_name '%s'. Defaulting to Phase 6 Joint Study.", exp_name)
        run_phase6_joint_study(str(config_path))


if __name__ == "__main__":
    main()
