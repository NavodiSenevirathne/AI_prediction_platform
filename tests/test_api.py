from src.config import FEATURES
from src.data.generate import generate_customers


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] == 1


def test_predict(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] == 1


def test_predict_rejects_invalid_input(client, customer):
    assert client.post("/predict", json={**customer, "age": 5}).status_code == 422
    assert client.post("/predict", json={**customer, "contract_type": "weekly"}).status_code == 422
    missing = {k: v for k, v in customer.items() if k != "tenure_months"}
    assert client.post("/predict", json=missing).status_code == 422


def test_high_risk_customer_scores_higher_than_loyal_one(client, customer):
    loyal = {
        **customer,
        "tenure_months": 60,
        "contract_type": "two_year",
        "num_support_calls": 0,
        "payment_method": "credit_card",
        "internet_service": "dsl",
        "monthly_charges": 50.0,
        "total_charges": 3000.0,
    }
    risky = client.post("/predict", json=customer).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_batch_and_monitoring(client):
    customers = generate_customers(120, drift=0.0, seed=1)[FEATURES].to_dict("records")
    r = client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    assert r.json()["count"] == 120

    summary = client.get("/monitoring/summary").json()
    assert summary["total_predictions"] == 120
    assert summary["window_predictions"] == 120
    assert summary["avg_latency_ms"] is not None

    drift = client.get("/monitoring/drift", params={"window": 100}).json()
    assert drift["status"] == "ok"
    assert set(drift["features"]) == set(FEATURES)

    assert len(client.get("/monitoring/recent", params={"limit": 10}).json()) == 10
    metrics = client.get("/metrics").text
    assert "predictions_total 120" in metrics


def test_drift_endpoint_needs_data(client):
    assert client.get("/monitoring/drift").json()["status"] == "insufficient_data"


def test_drifted_traffic_is_detected(client):
    customers = generate_customers(600, drift=1.0, seed=2)[FEATURES].to_dict("records")
    for i in range(0, 600, 200):
        client.post("/predict/batch", json={"customers": customers[i : i + 200]})
    drift = client.get("/monitoring/drift", params={"window": 600}).json()
    assert drift["dataset_drift"] is True


def test_model_reload_requires_token(client, monkeypatch):
    assert client.post("/model/reload").status_code == 403
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    r = client.post("/model/reload", headers={"x-admin-token": "secret"})
    assert r.status_code == 200
