import json
import shutil

import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data.generate import generate_customers
from src.monitoring.drift import detect_drift, psi_categorical, psi_numeric
from src.pipelines.retrain import drift_level_for, run_pipeline, validate
from src.training.model import evaluate, load_metadata, train


# ------------------------------------------------------------------ data
def test_generator_schema_and_reproducibility():
    a = generate_customers(500, seed=1)
    b = generate_customers(500, seed=1)
    assert list(a.columns) == FEATURES + [TARGET]
    pd.testing.assert_frame_equal(a, b)
    assert a[TARGET].isin([0, 1]).all()
    assert 0.1 < a[TARGET].mean() < 0.6


def test_generator_rejects_bad_drift():
    with pytest.raises(ValueError):
        generate_customers(10, drift=1.5)


def test_validate_catches_problems():
    df = generate_customers(1000, seed=3)
    assert validate(df) == []
    bad = df.copy()
    bad.loc[0, "contract_type"] = "weekly"
    assert any("unknown categories" in p for p in validate(bad))
    assert any("missing columns" in p for p in validate(df.drop(columns=["age"])))


# ------------------------------------------------------------------ drift
def test_psi_is_small_for_same_distribution_and_large_for_shift():
    ref = generate_customers(4000, drift=0.0, seed=1)
    same = generate_customers(4000, drift=0.0, seed=2)
    shifted = generate_customers(4000, drift=1.0, seed=3)
    assert psi_numeric(ref["monthly_charges"], same["monthly_charges"]) < 0.1
    assert psi_numeric(ref["monthly_charges"], shifted["monthly_charges"]) > 0.2
    assert psi_categorical(ref["contract_type"], same["contract_type"]) < 0.1


def test_detect_drift_report():
    ref = generate_customers(3000, drift=0.0, seed=1)
    assert detect_drift(ref, generate_customers(3000, drift=0.0, seed=2))["dataset_drift"] is False
    report = detect_drift(ref, generate_customers(3000, drift=1.0, seed=2))
    assert report["dataset_drift"] is True
    json.dumps(report)  # must be JSON serialisable


# ------------------------------------------------------------------ model
def test_model_quality_gate():
    df = generate_customers(6000, seed=5)
    model = train(df.iloc[:4500], {"max_iter": 100})
    metrics = evaluate(model, df.iloc[4500:])
    assert metrics["roc_auc"] > 0.8, metrics


# ------------------------------------------------------------------ pipeline
def test_drift_level_schedule():
    from datetime import date

    assert drift_level_for(date(2026, 10, 1)) == 0.0
    assert drift_level_for(date(2026, 10, 2)) == 0.0
    assert drift_level_for(date(2027, 1, 1)) == 1.0


def test_pipeline_keeps_champion_without_drift(model_dir, tmp_path):
    work = tmp_path / "models"
    shutil.copytree(model_dir, work)
    report = run_pipeline(
        drift=0.0, n_samples=2000, seed=11, model_dir=work, reports_dir=tmp_path / "r"
    )
    assert report["drift"]["dataset_drift"] is False
    assert (tmp_path / "r" / "latest_retrain_report.json").exists()


def test_pipeline_retrains_on_drift(model_dir, tmp_path):
    work = tmp_path / "models"
    shutil.copytree(model_dir, work)
    report = run_pipeline(
        drift=1.0, n_samples=4000, seed=12, model_dir=work, reports_dir=tmp_path / "r"
    )
    assert report["retrain_triggered"] is True
    assert "data drift" in report["trigger_reasons"]
    assert report["challenger_metrics"] is not None
    if report["promoted"]:
        assert load_metadata(work)["version"] == 2
