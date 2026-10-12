"""Data-drift detection.

For every feature we compare the *reference* data (what the current model was
trained on) with *current* data (recent production traffic) using:

* PSI - Population Stability Index (main decision metric)
    < 0.1  no significant change
    0.1-0.2 moderate change
    >= 0.2 significant drift  -> feature flagged as drifted
* A statistical test reported alongside PSI:
    numeric features     -> two-sample Kolmogorov-Smirnov test
    categorical features -> chi-squared test on category counts

The dataset is flagged as drifted when the share of drifted features is at
least ``DATASET_DRIFT_SHARE`` (default 30 %).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_DRIFT_SHARE,
    NUMERIC_FEATURES,
    PSI_DRIFT_THRESHOLD,
)

_EPS = 1e-4


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if reference.empty or current.empty:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature
        edges = np.unique(np.concatenate([edges, [reference.min() - 1, reference.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)
    return _psi_from_proportions(ref_counts, cur_counts)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi_from_proportions(ref, cur)


def _chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    table = np.array(
        [
            reference.value_counts().reindex(categories, fill_value=0).to_numpy(),
            current.value_counts().reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
    dataset_drift_share: float = DATASET_DRIFT_SHARE,
) -> dict[str, Any]:
    """Compare two datasets feature-by-feature and return a JSON-serialisable report."""
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = (
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )

    features: dict[str, dict[str, Any]] = {}
    for col in numeric_features:
        psi = psi_numeric(reference[col], current[col])
        p_value = float(stats.ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue)
        features[col] = {
            "type": "numeric",
            "psi": round(psi, 4),
            "test": "ks",
            "p_value": round(p_value, 6),
            "reference_mean": round(float(reference[col].mean()), 4),
            "current_mean": round(float(current[col].mean()), 4),
            "drifted": bool(psi >= psi_threshold),
        }
    for col in categorical_features:
        psi = psi_categorical(reference[col], current[col])
        features[col] = {
            "type": "categorical",
            "psi": round(psi, 4),
            "test": "chi2",
            "p_value": round(_chi2_pvalue(reference[col], current[col]), 6),
            "drifted": bool(psi >= psi_threshold),
        }

    n_drifted = sum(f["drifted"] for f in features.values())
    share = n_drifted / max(len(features), 1)
    return {
        "dataset_drift": bool(share >= dataset_drift_share),
        "share_drifted_features": round(share, 4),
        "n_drifted_features": n_drifted,
        "n_features": len(features),
        "psi_threshold": psi_threshold,
        "dataset_drift_share_threshold": dataset_drift_share,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": features,
    }
