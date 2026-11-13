"""
FairFed Risk - Data Ingestion Module
Downloads raw open-access clinical benchmark datasets:
1. UCI Heart Disease (Cleveland, Hungarian, Switzerland, VA Long Beach)
2. Pima Indians Diabetes Database
3. UCI Chronic Kidney Disease (Apollo Hospitals)
"""

from __future__ import annotations
import os
import urllib.request
import logging
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")

HEART_DISEASE_URLS = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

PIMA_DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def download_heart_disease(raw_dir: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Download 4 UCI Heart Disease sub-cohorts."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}
    for site_name, url in HEART_DISEASE_URLS.items():
        dest_path = raw_dir / f"heart_{site_name}.data"
        if not dest_path.exists():
            logger.info("Downloading UCI Heart Disease (%s) from %s...", site_name, url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                dest_path.write_bytes(content)
            logger.info("Saved %s (%d bytes)", dest_path, len(content))
        else:
            logger.info("UCI Heart Disease (%s) already exists at %s", site_name, dest_path)
        downloaded_paths[site_name] = dest_path
    return downloaded_paths


def download_pima_diabetes(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Download Pima Indians Diabetes dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / "pima_diabetes.csv"
    if not dest_path.exists():
        logger.info("Downloading Pima Indians Diabetes from %s...", PIMA_DIABETES_URL)
        req = urllib.request.Request(PIMA_DIABETES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            dest_path.write_bytes(content)
        logger.info("Saved %s (%d bytes)", dest_path, len(content))
    else:
        logger.info("Pima Diabetes already exists at %s", dest_path)
    return dest_path


def download_ckd_dataset(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Download UCI Chronic Kidney Disease via OpenML and save as CSV."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / "chronic_kidney_disease.csv"
    if not dest_path.exists():
        logger.info("Fetching Chronic Kidney Disease from OpenML (dataset 336)...")
        ckd_data = fetch_openml(name="chronic-kidney-disease", version=1, as_frame=True)
        df = ckd_data.frame
        df.to_csv(dest_path, index=False)
        logger.info("Saved CKD dataset (%d rows, %d cols) to %s", len(df), len(df.columns), dest_path)
    else:
        logger.info("CKD dataset already exists at %s", dest_path)
    return dest_path


def download_all_datasets(raw_dir: Path = RAW_DATA_DIR) -> dict[str, Path | dict[str, Path]]:
    """Download all required datasets for the FairFed Risk research project."""
    logger.info("Starting ingestion of public benchmark clinical datasets...")
    heart_paths = download_heart_disease(raw_dir)
    pima_path = download_pima_diabetes(raw_dir)
    ckd_path = download_ckd_dataset(raw_dir)
    logger.info("All benchmark datasets successfully verified and saved.")
    return {
        "heart": heart_paths,
        "pima": pima_path,
        "ckd": ckd_path,
    }


if __name__ == "__main__":
    download_all_datasets()
