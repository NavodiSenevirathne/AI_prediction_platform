"""Model definition, evaluation and the local model registry (``models/``).

The ``models/`` folder holds the *production* (champion) model that gets baked
into the Docker image:

    models/model.joblib          the fitted scikit-learn pipeline
    models/metadata.json         version, metrics, lineage and version history
    models/reference_sample.csv  sample of the training data, used as the
                                 baseline for drift detection

MLflow keeps the full experiment history and every candidate model; this
folder holds only the model that is currently promoted to production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CATEGORIES,
    FEATURES,
    METADATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    NUMERIC_FEATURES,
    REFERENCE_FILE,
    TARGET,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.06,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
}
DECISION_THRESHOLD = 0.5
HISTORY_LIMIT = 25


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    params = {**DEFAULT_PARAMS, **(params or {})}
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORIES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(**params)),
        ]
    )


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    y_true = df[TARGET].to_numpy()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))), 4),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def train(train_df: pd.DataFrame, params: dict[str, Any] | None = None) -> Pipeline:
    model = build_pipeline(params)
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


# --------------------------------------------------------------------------- registry
def load_metadata(model_dir: Path = MODEL_DIR) -> dict[str, Any] | None:
    path = Path(model_dir) / METADATA_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_production_model(model_dir: Path = MODEL_DIR) -> tuple[Pipeline, dict, pd.DataFrame]:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / MODEL_FILE)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    reference = pd.read_csv(model_dir / REFERENCE_FILE)
    return model, metadata, reference


def save_production_model(
    model: Pipeline,
    metrics: dict[str, float],
    reference: pd.DataFrame,
    params: dict[str, Any] | None = None,
    model_dir: Path = MODEL_DIR,
    extra: dict[str, Any] | None = None,
    reference_size: int = 3000,
) -> dict[str, Any]:
    """Promote a model to production: write model, metadata and reference sample."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    previous = load_metadata(model_dir)
    version = (previous["version"] + 1) if previous else 1
    history = list(previous.get("history", [])) if previous else []

    now = datetime.now(UTC).isoformat(timespec="seconds")
    metadata: dict[str, Any] = {
        "model_name": "churn-classifier",
        "version": version,
        "trained_at": now,
        "algorithm": "HistGradientBoostingClassifier",
        "features": FEATURES,
        "target": TARGET,
        "decision_threshold": DECISION_THRESHOLD,
        "params": {**DEFAULT_PARAMS, **(params or {})},
        "metrics": metrics,
        **(extra or {}),
    }
    history.append(
        {
            "version": version,
            "trained_at": now,
            "roc_auc": metrics.get("roc_auc"),
            "f1": metrics.get("f1"),
            "reason": (extra or {}).get("promotion_reason", "initial training"),
            "mlflow_run_id": (extra or {}).get("mlflow_run_id"),
        }
    )
    metadata["history"] = history[-HISTORY_LIMIT:]

    sample = reference.sample(min(reference_size, len(reference)), random_state=0)
    tmp_model = model_dir / f".{MODEL_FILE}.tmp"
    joblib.dump(model, tmp_model)
    tmp_model.replace(model_dir / MODEL_FILE)
    sample[FEATURES + [TARGET]].to_csv(model_dir / REFERENCE_FILE, index=False)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return metadata
