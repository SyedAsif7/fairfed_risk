"""
FairFed Risk - Federated Server Simulation Engine
Manages round-by-round federated training, communication, client sampling,
differential privacy accounting, and global evaluation across heterogeneous hospital nodes.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.utils.data

from src.models.multitask_mlp import MultiTaskRiskMLP
from src.federated.client import HospitalFlowerClient
from src.federated.strategies import aggregate_weights, evaluate_global_model
from src.evaluation.evaluator import evaluate_torch_model
from src.data.partition import HospitalData, get_centralized_dataset
from src.data.dataset import ClinicalMultiTaskDataset
from src.privacy.accountant import PrivacyAccountant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FederatedSimulationServer:
    """
    Federated Orchestrator for FairFed Risk experiments.
    Executes reproducible simulation across simulated clinical sites with Differential Privacy.
    """

    def __init__(
        self,
        hospital_partitions: Dict[int, HospitalData],
        strategy_name: str = "fedavg",  # "fedavg", "fedprox", "fairfed"
        num_rounds: int = 15,
        local_epochs: int = 2,
        batch_size: int = 32,
        lr: float = 0.003,
        mu: float = 0.0,  # FedProx proximal coefficient
        fairfed_beta: float = 0.0,  # FairFed aggregation step size
        fairness_penalty_lambda: float = 0.0,  # In-processing client loss penalty
        target_epsilon: float = float("inf"),  # DP target epsilon
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        device: torch.device = torch.device("cpu"),
        random_seed: int = 42,
    ):
        self.hospital_partitions = hospital_partitions
        self.strategy_name = strategy_name.lower()
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mu = float(mu)
        self.fairfed_beta = fairfed_beta
        self.fairness_penalty_lambda = fairness_penalty_lambda
        self.target_epsilon = target_epsilon
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.device = device
        self.random_seed = random_seed

        # Calibrate noise multiplier for target epsilon if private
        avg_train_n = float(np.mean([p.n_train for p in hospital_partitions.values()]))
        sample_rate = float(np.clip(batch_size / max(avg_train_n, 1.0), 0.01, 1.0))
        steps_per_client = num_rounds * local_epochs * int(np.ceil(avg_train_n / batch_size))

        if target_epsilon < float("inf") and target_epsilon > 0:
            self.noise_multiplier = PrivacyAccountant.calibrate_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                total_steps=steps_per_client,
            )
            logger.info(
                "Calibrated DP-SGD: Target eps=%.2f (delta=%.1e) -> noise_multiplier=%.4f (steps=%d)",
                target_epsilon,
                delta,
                self.noise_multiplier,
                steps_per_client,
            )
        else:
            self.noise_multiplier = 0.0

        # Global evaluation dataset assembled across all test partitions
        tr_x, tr_y, tr_m, tr_s, te_x, te_y, te_m, te_s = get_centralized_dataset(hospital_partitions)
        self.global_test_ds = ClinicalMultiTaskDataset(te_x, te_y, te_m, te_s)
        self.global_test_loader = torch.utils.data.DataLoader(
            self.global_test_ds, batch_size=batch_size, shuffle=False
        )

        # Global model initialization
        torch.manual_seed(random_seed)
        self.global_model = MultiTaskRiskMLP(input_dim=7, hidden_dim1=64, hidden_dim2=32).to(self.device)
        self.global_parameters = [val.detach().cpu().numpy().copy() for val in self.global_model.state_dict().values()]

        # Initialize clients
        self.clients: Dict[int, HospitalFlowerClient] = {}
        for hid, h_data in hospital_partitions.items():
            self.clients[hid] = HospitalFlowerClient(
                hospital_data=h_data,
                batch_size=batch_size,
                lr=lr,
                mu=self.mu,
                fairness_penalty_lambda=self.fairness_penalty_lambda,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                delta=delta,
                device=device,
            )

        # Initial client weights (sample proportion by default)
        total_train_samples = sum(p.n_train for p in hospital_partitions.values())
        self.client_sample_weights = {
            hid: p.n_train / total_train_samples for hid, p in hospital_partitions.items()
        }
        self.dynamic_client_weights = self.client_sample_weights.copy()

    def run(self) -> Dict[str, Any]:
        """Executes federated communication rounds."""
        logger.info(
            "Starting Federated Simulation: Strategy=%s | Target Eps=%s | Rounds=%d | mu=%.3f",
            self.strategy_name,
            str(self.target_epsilon),
            self.num_rounds,
            self.mu,
        )

        history: Dict[str, Any] = {
            "strategy": self.strategy_name,
            "target_epsilon": self.target_epsilon if self.target_epsilon < float("inf") else "inf",
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "rounds": [],
            "train_loss": [],
            "epsilon_spent": [],
            "global_auroc": [],
            "global_auprc": [],
            "global_brier": [],
            "sex_dpd": [],
            "sex_eod": [],
            "age_dpd": [],
            "age_eod": [],
            "round_metrics": [],
        }

        # Evaluate Round 0 baseline
        init_eval = evaluate_global_model(
            self.global_model, self.global_parameters, self.global_test_loader, device=self.device
        )
        logger.info(
            "Round 0 (Initial) -> Macro AUROC: %.4f | Macro AUPRC: %.4f | Brier: %.4f",
            init_eval["macro"]["auroc"],
            init_eval["macro"]["auprc"],
            init_eval["macro"]["brier"],
        )

        fit_config = {
            "local_epochs": self.local_epochs,
            "mu": self.mu,
            "lr": self.lr,
        }

        for r in range(1, self.num_rounds + 1):
            client_params_list = []
            client_weights_list = []
            client_losses = []
            client_epsilons = []
            client_eval_fairness: Dict[int, float] = {}

            # 1. Distribute global model to clients and train locally
            for hid, client in self.clients.items():
                updated_params, num_samples, fit_res = client.fit(
                    parameters=self.global_parameters,
                    config=fit_config,
                )
                client_params_list.append(updated_params)
                client_losses.append(fit_res["train_loss"])
                client_epsilons.append(fit_res["epsilon_spent"])

                # Determine aggregation weight
                if self.strategy_name == "fairfed" and r > 1:
                    weight = self.dynamic_client_weights[hid]
                else:
                    weight = self.client_sample_weights[hid]
                client_weights_list.append(weight)

                # Local validation evaluation for FairFed reweighting
                if self.strategy_name == "fairfed":
                    local_metrics = evaluate_torch_model(client._get_underlying_model(), client.test_loader, device=self.device)
                    local_dpd = local_metrics["fairness"]["by_sex"]["macro_dpd"]
                    client_eval_fairness[hid] = local_dpd

            # 2. Server Aggregation
            self.global_parameters = aggregate_weights(client_params_list, client_weights_list)

            # 3. FairFed Dynamic Weight Update
            if self.strategy_name == "fairfed" and client_eval_fairness:
                avg_fairness = float(np.mean(list(client_eval_fairness.values())))
                raw_weights = {}
                for hid in self.clients.keys():
                    f_diff = client_eval_fairness[hid] - avg_fairness
                    new_w = max(self.dynamic_client_weights[hid] - self.fairfed_beta * f_diff, 0.01)
                    raw_weights[hid] = new_w
                total_w = sum(raw_weights.values())
                self.dynamic_client_weights = {hid: w / total_w for hid, w in raw_weights.items()}

            # 4. Global Evaluation after round aggregation
            round_eval = evaluate_global_model(
                self.global_model, self.global_parameters, self.global_test_loader, device=self.device
            )

            avg_train_loss = float(np.mean(client_losses))
            max_eps_spent = float(np.max(client_epsilons)) if client_epsilons else float("inf")
            macro_auc = round_eval["macro"]["auroc"]
            macro_ap = round_eval["macro"]["auprc"]
            brier = round_eval["macro"]["brier"]
            sex_dpd = round_eval["fairness"]["by_sex"]["macro_dpd"]
            sex_eod = round_eval["fairness"]["by_sex"]["macro_eod"]
            age_dpd = round_eval["fairness"]["by_age"]["macro_dpd"]
            age_eod = round_eval["fairness"]["by_age"]["macro_eod"]

            history["rounds"].append(r)
            history["train_loss"].append(avg_train_loss)
            history["epsilon_spent"].append(max_eps_spent)
            history["global_auroc"].append(macro_auc)
            history["global_auprc"].append(macro_ap)
            history["global_brier"].append(brier)
            history["sex_dpd"].append(sex_dpd)
            history["sex_eod"].append(sex_eod)
            history["age_dpd"].append(age_dpd)
            history["age_eod"].append(age_eod)
            history["round_metrics"].append(round_eval)

            if r % 3 == 0 or r == self.num_rounds:
                logger.info(
                    "Round %2d/%2d -> Loss: %.4f | Eps: %s | AUROC: %.4f | AUPRC: %.4f | Sex DPD: %.4f",
                    r,
                    self.num_rounds,
                    avg_train_loss,
                    f"{max_eps_spent:.2f}" if max_eps_spent < float("inf") else "inf",
                    macro_auc,
                    macro_ap,
                    sex_dpd,
                )

        history["final_metrics"] = history["round_metrics"][-1]
        history["final_parameters"] = self.global_parameters
        return history
