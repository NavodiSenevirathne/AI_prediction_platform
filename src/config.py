"""Central configuration shared by training, the API and the pipelines."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT_DIR / "reports"))

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference_sample.csv"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "avg_daily_usage_gb",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

CATEGORIES = {
    "contract_type": ["month_to_month", "one_year", "two_year"],
    "internet_service": ["dsl", "fiber", "none"],
    "payment_method": ["credit_card", "bank_transfer", "electronic_check"],
}

MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction-platform")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churn-classifier")

# Drift / retraining policy
PSI_DRIFT_THRESHOLD = float(os.getenv("PSI_DRIFT_THRESHOLD", "0.2"))
DATASET_DRIFT_SHARE = float(os.getenv("DATASET_DRIFT_SHARE", "0.3"))
MIN_ACCEPTABLE_AUC = float(os.getenv("MIN_ACCEPTABLE_AUC", "0.80"))
MAX_AUC_DROP = float(os.getenv("MAX_AUC_DROP", "0.03"))
