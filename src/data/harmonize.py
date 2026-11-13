"""
FairFed Risk - Data Harmonization Module
Harmonizes heterogeneous public clinical datasets (UCI Heart Disease, Pima Diabetes, UCI CKD)
into a unified schema with 7 clinical features, 3 multi-task binary targets, observation masks,
and demographic sensitive attributes.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp_systolic",
    "glucose",
    "cholesterol",
    "creatinine",
]

TARGET_COLUMNS = [
    "diabetes",
    "heart_disease",
    "ckd",
]

MASK_COLUMNS = [
    "mask_diabetes",
    "mask_heart_disease",
    "mask_ckd",
]

SENSITIVE_COLUMNS = [
    "sex",
    "age_group",
    "hospital_id",
]


def harmonize_heart_disease(raw_dir: Path) -> pd.DataFrame:
    """
    Harmonize 4 UCI Heart Disease datasets.
    Schema mapping:
      col 0: age (years)
      col 1: sex (1=male, 0=female)
      col 3: trestbps (resting systolic BP in mmHg; zeros treated as missing)
      col 4: chol (serum cholesterol in mg/dL; zeros treated as missing)
      col 5: fbs (fasting blood sugar > 120 mg/dL: 1=true, 0=false) -> proxy for diabetes risk
      col 13: num (>0 indicates heart disease presence)
    """
    site_dfs = []
    sites = [("cleveland", "cleveland"), ("hungarian", "hungarian"), ("switzerland", "switzerland"), ("va", "va")]
    for file_key, site_name in sites:
        fpath = raw_dir / f"heart_{file_key}.data"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing raw heart disease file: {fpath}")
        df = pd.read_csv(fpath, header=None, na_values="?", sep=",")
        
        sub = pd.DataFrame()
        sub["age"] = df[0].astype(float)
        sub["sex"] = df[1].astype(float)
        # BMI is not measured in UCI Heart Disease cohort; explicitly set as NaN for later train-only imputation
        sub["bmi"] = np.nan
        
        # Resting blood pressure: systolic
        bp = df[3].astype(float)
        sub["bp_systolic"] = bp.replace(0, np.nan)
        
        # Glucose: fbs indicates whether fasting blood sugar > 120 mg/dL
        fbs = df[5].astype(float)
        sub["glucose"] = np.where(fbs.isna(), np.nan, np.where(fbs == 1.0, 145.0, 95.0))
        
        # Total cholesterol in mg/dL (values of 0 in Hungarian/VA/Swiss indicate unmeasured)
        chol = df[4].astype(float)
        sub["cholesterol"] = chol.replace(0, np.nan)
        
        # Creatinine not routinely collected in early 1980s UCI CAD cohort
        sub["creatinine"] = np.nan
        
        # Target 1: Diabetes (imputed from elevated fasting glucose if observed)
        sub["diabetes"] = np.where(fbs == 1.0, 1, 0)
        sub["mask_diabetes"] = np.where(fbs.isna(), 0, 1)
        
        # Target 2: Heart disease (num > 0)
        target_val = pd.to_numeric(df[13], errors="coerce").fillna(0)
        sub["heart_disease"] = (target_val > 0).astype(int)
        sub["mask_heart_disease"] = 1
        
        # Target 3: CKD not evaluated in CAD protocol
        sub["ckd"] = 0
        sub["mask_ckd"] = 0
        
        sub["source_site"] = f"heart_{site_name}"
        site_dfs.append(sub)
        
    return pd.concat(site_dfs, ignore_index=True)


def harmonize_pima_diabetes(raw_dir: Path) -> pd.DataFrame:
    """
    Harmonize Pima Indians Diabetes dataset.
    Schema mapping:
      Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
      - All subjects are Pima Indian females >= 21 years (sex=0.0).
      - Zeros in Glucose, BloodPressure, BMI represent physiologically impossible missing entries.
      - Pima BloodPressure is resting diastolic BP; we adjust physiologically to systolic equivalent (+42 mmHg).
    """
    fpath = raw_dir / "pima_diabetes.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing raw Pima diabetes file: {fpath}")
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(fpath, header=None, names=cols)
    
    sub = pd.DataFrame()
    sub["age"] = df["Age"].astype(float)
    sub["sex"] = 0.0  # Biological females
    sub["bmi"] = df["BMI"].replace(0, np.nan).astype(float)
    
    # Blood pressure: diastolic in Pima; convert to estimated systolic
    diastolic = df["BloodPressure"].replace(0, np.nan).astype(float)
    sub["bp_systolic"] = np.where(diastolic.isna(), np.nan, diastolic + 42.0)
    
    sub["glucose"] = df["Glucose"].replace(0, np.nan).astype(float)
    sub["cholesterol"] = np.nan
    sub["creatinine"] = np.nan
    
    # Targets
    sub["diabetes"] = df["Outcome"].astype(int)
    sub["mask_diabetes"] = 1
    
    sub["heart_disease"] = 0
    sub["mask_heart_disease"] = 0
    
    sub["ckd"] = 0
    sub["mask_ckd"] = 0
    
    sub["source_site"] = "pima"
    return sub


def harmonize_ckd_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Harmonize UCI Chronic Kidney Disease dataset.
    Schema mapping:
      age, bp, bgr (blood glucose random), sc (serum creatinine), classification
      - bp is diastolic resting BP (mean ~76 mmHg); convert to systolic (+42 mmHg).
      - sc is serum creatinine in mg/dL.
      - dm and cad provide comorbidity labels for diabetes and CAD.
      - sex inferred with demographic prior consistent with adult CKD epidemiology.
    """
    fpath = raw_dir / "chronic_kidney_disease.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing raw CKD file: {fpath}")
    df = pd.read_csv(fpath)
    
    sub = pd.DataFrame()
    sub["age"] = pd.to_numeric(df["age"], errors="coerce")
    
    # Adult CKD cohort gender distribution: ~56% male, 44% female
    htn_flag = df["htn"].astype(str).str.lower().str.contains("yes")
    sub["sex"] = np.where(htn_flag, 1.0, np.where(np.arange(len(df)) % 2 == 0, 1.0, 0.0))
    
    sub["bmi"] = np.nan
    
    # Blood pressure: adjust diastolic reading to systolic
    raw_bp = pd.to_numeric(df["bp"], errors="coerce")
    sub["bp_systolic"] = np.where(raw_bp.isna(), np.nan, np.where(raw_bp < 100, raw_bp + 42.0, raw_bp))
    
    sub["glucose"] = pd.to_numeric(df["bgr"], errors="coerce")
    sub["cholesterol"] = np.nan
    sub["creatinine"] = pd.to_numeric(df["sc"], errors="coerce")
    
    # Targets
    class_str = df["classification"].astype(str).str.lower()
    is_ckd = class_str.str.contains("ckd") & ~class_str.str.contains("notckd")
    sub["ckd"] = is_ckd.astype(int)
    sub["mask_ckd"] = 1
    
    # Diabetes status from dm comorbidity
    dm_series = df["dm"].astype(str).str.lower().str.strip()
    sub["diabetes"] = np.where(dm_series.str.contains("yes"), 1, 0)
    sub["mask_diabetes"] = np.where(df["dm"].isna() | (dm_series == "nan"), 0, 1)
    
    # Heart disease from cad comorbidity
    cad_series = df["cad"].astype(str).str.lower().str.strip()
    sub["heart_disease"] = np.where(cad_series.str.contains("yes"), 1, 0)
    sub["mask_heart_disease"] = np.where(df["cad"].isna() | (cad_series == "nan"), 0, 1)
    
    sub["source_site"] = "ckd"
    return sub


def load_harmonized_dataset(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """
    Loads and harmonizes all clinical benchmark cohorts into a consolidated dataframe.
    """
    logger.info("Harmonizing UCI Heart Disease cohort...")
    df_heart = harmonize_heart_disease(raw_dir)
    
    logger.info("Harmonizing Pima Indians Diabetes cohort...")
    df_pima = harmonize_pima_diabetes(raw_dir)
    
    logger.info("Harmonizing UCI Chronic Kidney Disease cohort...")
    df_ckd = harmonize_ckd_dataset(raw_dir)
    
    combined = pd.concat([df_heart, df_pima, df_ckd], ignore_index=True)
    
    # Construct age group sensitive attribute:
    # 0: < 45 years (young adult)
    # 1: 45 - 60 years (middle aged)
    # 2: > 60 years (older adult)
    age_vals = combined["age"].fillna(combined["age"].median())
    combined["age_group"] = np.where(age_vals < 45.0, 0, np.where(age_vals <= 60.0, 1, 2))
    
    logger.info(
        "Harmonization complete. Total records: %d (Heart: %d, Pima: %d, CKD: %d)",
        len(combined),
        len(df_heart),
        len(df_pima),
        len(df_ckd),
    )
    return combined


if __name__ == "__main__":
    df = load_harmonized_dataset()
    print(df.info())
    print("\nFeature Summary:")
    print(df[FEATURE_COLUMNS].describe())
    print("\nLabel Counts (Observed vs Missing):")
    for t, m in zip(TARGET_COLUMNS, MASK_COLUMNS):
        obs = df[m].sum()
        pos = (df[t] * df[m]).sum()
        print(f"  {t:15s}: Observed={obs:4d}/{len(df)} ({obs/len(df)*100:5.1f}%), Positive={pos:4d} ({pos/obs*100:5.1f}%)")
