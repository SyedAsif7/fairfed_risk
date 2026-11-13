# FairFed Risk

> **Privacy-Preserving Federated Learning for Multi-Disease Early Risk Prediction under Subgroup Fairness Constraints**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.4%2B-pink.svg)](https://flower.ai/)
[![Opacus](https://img.shields.io/badge/Opacus-1.4%2B-green.svg)](https://opacus.ai/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.9%2B-yellow.svg)](https://fairlearn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

*Research Prototype Disclaimer: FairFed Risk is a clinical research prototype evaluated on open-access benchmark cohorts (UCI Heart Disease, Pima Indians Diabetes, UCI Chronic Kidney Disease). It is designed for computational research on privacy and fairness dynamics in healthcare federated learning and has not been clinically validated for direct patient triage or diagnostic prescription.*

---

## Overview

Clinical predictive models trained on pooled multi-institution health records frequently encounter severe barriers:
1. **Privacy & Regulatory Walls**: Patient confidentiality (HIPAA / GDPR) prohibits aggregating raw electronic health records (EHR) into a centralized repository.
2. **Specialized Institutional Siloing**: Individual hospital centers specialize in specific diseases (e.g., cardiac catheterization clinics vs. nephrology dialysis centers vs. primary care diabetes clinics). Siloed local models fail to generalize across multimorbid conditions.
3. **Privacy-Fairness Conflicts**: Differential privacy noise ($\epsilon, \delta$) added to gradient updates disproportionately degrades predictive utility for demographic minority subgroups (e.g., younger cohorts, biological females in cardiovascular settings).

**FairFed Risk** provides an end-to-end framework integrating:
- **Harmonized Multi-Task Learning**: Predicts Type 2 Diabetes, Coronary Heart Disease, and Chronic Kidney Disease simultaneously with masked loss for unobserved tests.
- **Federated Simulation**: Non-IID clinical simulation across 6 realistic hospital institutions using Flower (`flwr`) with FedAvg and FedProx.
- **Formal Differential Privacy (DP-SGD)**: Per-sample gradient clipping and calibrated Gaussian noise via Opacus with cumulative privacy accounting ($\epsilon \in \{1, 3, 8, \infty\}$, $\delta = 10^{-5}$).
- **Dual Fairness Mitigations**: In-processing client covariance loss penalty and server-side dynamic client aggregation reweighting (FairFed).
- **Interactive Streamlit Auditing Dashboard**: Real-time privacy spending tracker, subgroup fairness diagnostics, and cross-hospital comparison matrix.

---

## Project Structure

```
fairfed-risk/
  README.md
  requirements.txt
  configs/            # Experiment YAML configurations
  data/               # Ingestion and preprocessing scripts
    raw/              # Raw benchmark datasets
    processed/        # Harmonized multi-task clinical records
  src/
    data/             # Ingestion, harmonization, non-IID partitioning
    models/           # Multi-task risk prediction MLP trunk & heads
    federated/        # Flower client, server, and aggregation strategies
    privacy/          # Opacus DP-SGD engine and privacy accountant
    fairness/         # Subgroup metrics, penalty loss, dynamic reweighting
    evaluation/       # AUROC, AUPRC, calibration, fairness diagnostics
  experiments/        # Runnable phase-by-phase experiment drivers
  dashboard/          # Interactive Streamlit auditing interface
  tests/              # Comprehensive pytest test suite
  results/            # Saved metrics, logs, and model checkpoints
  report/             # Scientific whitepaper draft
```

---

## Installation & Setup

```bash
git clone https://github.com/SyedAsif7/fairfed_risk.git
cd fairfed_risk
pip install -r requirements.txt
```

---

## Empirical Benchmark Progress

### Baselines vs. Federated Learning (Global Held-Out Multi-Center Test Set, $N=419$)

| Architecture / Strategy | Training Modality | Macro AUROC | Macro AUPRC | Brier Score | Sex DPD | Age DPD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Pooled (Upper Bound)** | Centralized ($N=1,669$) | **0.9143** | **0.8778** | **0.1150** | 0.3359 | 0.2812 |
| **Local-Only (Hospital 0 - Cleveland)**| Siloed ($N=242$) | 0.6689 | 0.6691 | 0.2060 | 0.1735 | 0.2765 |
| **Local-Only (Hospital 3 - Apollo)**   | Siloed ($N=240$) | 0.7804 | 0.7375 | 0.2132 | 0.1812 | 0.4694 |
| **Local-Only (Hospital 4 - Pima)**     | Siloed ($N=440$) | 0.6155 | 0.6092 | 0.2369 | 0.0788 | 0.1829 |
| **Federated: FedAvg**                  | 6 Hospitals ($R=15$) | 0.7572 | 0.7503 | 0.1836 | 0.1882 | 0.4950 |
| **Federated: FedProx ($\mu=0.01$)**    | 6 Hospitals ($R=15$) | **0.8003** | **0.7655** | **0.1582** | 0.2994 | 0.5331 |

---

## Citation & Acknowledgments
This work builds upon foundational principles in federated optimization and algorithmic fairness:
- **FairFed**: Ezzeldin et al., *"FairFed: Fairness-Aware Federated Learning via Local Loss and Aggregation Reweighting"* (2021).
- **FedAvg**: McMahan et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"* (AISTATS 2017).
- **FedProx**: Li et al., *"Federated Optimization in Heterogeneous Networks"* (MLSys 2020).
- **DP-SGD**: Abadi et al., *"Deep Learning with Differential Privacy"* (CCS 2016).
