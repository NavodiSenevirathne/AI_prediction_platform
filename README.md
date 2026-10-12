# Real-Time AI Prediction Platform with Automated MLOps Pipeline

A production-style machine learning platform that serves **real-time customer-churn
predictions**, **monitors data drift** on live traffic, **retrains itself automatically**
when the data changes, and **redeploys the new model through CI/CD** with no manual steps.

![Architecture](docs/architecture.png)

| | |
|---|---|
| **Live API** | `https://<your-api>.onrender.com/docs` |
| **Live dashboard** | `https://<your-dashboard>.onrender.com` |
| **Experiment tracking** | `https://dagshub.com/<user>/mlops-prediction-platform.mlflow` |

> Replace the links above after deployment (see [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)).

---

## Business scenario

A telecom company wants to know, the moment a customer interacts with any channel, how
likely that customer is to cancel (churn), so retention offers can be targeted in real
time. Customer behaviour changes over time (new pricing, more fibre customers, heavier
usage), so a model trained once goes stale. The platform has to **detect that change and
update the model automatically**.

## What the platform does

| Requirement | Implementation |
|---|---|
| Real-time predictions | FastAPI service, `POST /predict` and `/predict/batch`, ~10 ms per request |
| Automated retraining | Daily GitHub Actions workflow: ingest → validate → drift check → train challenger → champion-vs-challenger → promote |
| Continuous deployment | CD workflow builds versioned Docker images, pushes to GHCR, triggers Render deploy hooks and verifies the new model version is live |
| Experiment tracking | MLflow (hosted on DagsHub): params, metrics, drift reports, model artifacts, Model Registry |
| Data drift monitoring | PSI + Kolmogorov–Smirnov / χ² tests per feature, live on production traffic (`/monitoring/drift`) and inside the retraining pipeline |
| Cloud deployment | Two Docker web services on Render (API + dashboard), defined in `render.yaml` |
| CI/CD integration | `ci.yml` (lint, tests, quality gate, image build, container smoke test), `cd.yml`, `retrain.yml` |
| Monitoring dashboard | Streamlit: traffic, latency, churn rate, PSI per feature, model version history, live prediction form, traffic simulator |

## Repository structure

```
├── src/
│   ├── config.py                 # features, paths, drift/retraining thresholds
│   ├── data/generate.py          # data source: synthetic customers with controllable drift
│   ├── training/
│   │   ├── model.py              # sklearn pipeline, evaluation, production model registry
│   │   ├── tracking.py           # MLflow integration (safe fallback when offline)
│   │   └── train.py              # initial (bootstrap) training
│   ├── monitoring/drift.py       # PSI, KS and chi-squared drift detection
│   ├── pipelines/retrain.py      # automated champion/challenger retraining pipeline
│   └── api/                      # FastAPI app, schemas, prediction log (SQLite)
├── dashboard/app.py              # Streamlit monitoring dashboard
├── scripts/simulate_traffic.py   # streams simulated real-time traffic to the API
├── models/                       # current production model + metadata + reference data
├── reports/                      # retraining reports (written by the pipeline)
├── tests/                        # 17 unit + integration tests
├── .github/workflows/            # ci.yml, cd.yml, retrain.yml
├── Dockerfile                    # API image
├── Dockerfile.dashboard          # dashboard image
├── docker-compose.yml            # API + dashboard + MLflow locally
├── render.yaml                   # cloud deployment blueprint
└── docs/                         # architecture, technical documentation, setup guide
```

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # run the test suite
uvicorn src.api.main:app --reload        # API on http://localhost:8000/docs
```

In a second terminal:

```bash
python scripts/simulate_traffic.py --n 300                 # normal traffic
python scripts/simulate_traffic.py --n 300 --drift 0.9     # drifted traffic
streamlit run dashboard/app.py                             # dashboard on http://localhost:8501
```

Retraining and experiment tracking:

```bash
python -m src.pipelines.retrain --drift 0.8   # drift-aware retraining
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Or everything with Docker:

```bash
docker compose up --build        # API :8000, dashboard :8501, MLflow :5000
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Score one customer |
| POST | `/predict/batch` | Score up to 1000 customers |
| GET | `/health` | Liveness + model version |
| GET | `/model/info` | Model metadata, metrics, version history |
| POST | `/model/reload` | Hot-reload model (needs `X-Admin-Token`) |
| GET | `/monitoring/summary` | Traffic, latency, prediction distribution |
| GET | `/monitoring/drift` | Live drift report on the last N predictions |
| GET | `/monitoring/recent` | Latest logged predictions |
| GET | `/metrics` | Prometheus metrics |

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 34, "tenure_months": 5, "monthly_charges": 89.5, "total_charges": 447.5,
  "num_support_calls": 4, "avg_daily_usage_gb": 6.2, "contract_type": "month_to_month",
  "internet_service": "fiber", "payment_method": "electronic_check"}'
```

```json
{"prediction_id": "774a…", "churn_probability": 0.9021, "churn_prediction": true,
 "risk_level": "high", "model_version": 1, "latency_ms": 9.8}
```

## The automated MLOps loop

1. Customers hit the **API**; every prediction is logged.
2. The **dashboard** and `/monitoring/drift` compare live traffic with the model's training data.
3. Every day, the **retraining workflow** pulls the latest labelled data, checks drift and model
   performance, trains a challenger, and promotes it only if it beats the production model.
4. A promoted model is committed to `models/` and the **CD workflow** builds a new image
   (`api:model-vN`), deploys it to Render and waits until the live API reports version N.
5. Every run is recorded in **MLflow**, and the model is registered in the Model Registry.

Try it: **GitHub → Actions → Automated retraining → Run workflow** with drift `0.8`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Technical documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Setup & deployment guide](docs/SETUP_GUIDE.md)

## Tech stack

Python 3.12 · scikit-learn · FastAPI · Streamlit · MLflow · DagsHub · Docker ·
GitHub Actions · GitHub Container Registry · Render · pytest · ruff
