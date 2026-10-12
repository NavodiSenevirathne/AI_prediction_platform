import os

import pytest

os.environ.setdefault("DISABLE_MLFLOW", "1")

from src.data.generate import generate_customers  # noqa: E402
from src.training.model import save_production_model, train  # noqa: E402


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A small trained production model in a temporary folder."""
    path = tmp_path_factory.mktemp("models")
    df = generate_customers(3000, drift=0.0, seed=0)
    model = train(df, {"max_iter": 60})
    save_production_model(model, {"roc_auc": 0.85, "f1": 0.6}, reference=df, model_dir=path)
    return path


@pytest.fixture
def client(model_dir, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    app = create_app(model_dir=model_dir, db_path=tmp_path / "predictions.db")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def customer():
    return {
        "age": 34,
        "tenure_months": 5,
        "monthly_charges": 89.5,
        "total_charges": 447.5,
        "num_support_calls": 4,
        "avg_daily_usage_gb": 6.2,
        "contract_type": "month_to_month",
        "internet_service": "fiber",
        "payment_method": "electronic_check",
    }
