"""Real-time churn prediction API.

Run locally:
    uvicorn src.api.main:app --reload
Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.api.prediction_store import PredictionStore
from src.api.schemas import BatchRequest, BatchResponse, CustomerFeatures, PredictionResponse
from src.config import DATA_DIR, FEATURES, MODEL_DIR
from src.monitoring.drift import detect_drift
from src.training.model import load_production_model

API_VERSION = "1.0.0"


def _risk_level(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"


class ModelService:
    """Holds the production model in memory and makes predictions."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.load()

    def load(self) -> None:
        self.model, self.metadata, self.reference = load_production_model(self.model_dir)
        self.loaded_at = datetime.now(UTC).isoformat(timespec="seconds")

    @property
    def version(self) -> int:
        return int(self.metadata["version"])

    def predict(self, customers: list[CustomerFeatures]) -> list[float]:
        frame = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
        return self.model.predict_proba(frame)[:, 1].tolist()


def create_app(model_dir: Path | None = None, db_path: Path | None = None) -> FastAPI:
    model_dir = Path(model_dir or MODEL_DIR)
    db_path = Path(db_path or os.getenv("PREDICTIONS_DB", DATA_DIR / "predictions.db"))
    state: dict[str, Any] = {"last_drift": None, "started_at": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = ModelService(model_dir)
        app.state.store = PredictionStore(db_path)
        yield
        app.state.store.close()

    app = FastAPI(
        title="Real-Time Churn Prediction Platform",
        description=(
            "Real-time ML predictions with logging, drift monitoring and "
            "automated retraining (MLOps demo platform)."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def _predict_and_log(customers: list[CustomerFeatures]) -> tuple[list[PredictionResponse], float]:
        service: ModelService = app.state.model
        start = time.perf_counter()
        probabilities = service.predict(customers)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        threshold = service.metadata.get("decision_threshold", 0.5)
        per_item = round(latency_ms / len(customers), 3)
        now = datetime.now(UTC).isoformat()

        responses, rows = [], []
        for customer, p in zip(customers, probabilities, strict=False):
            pid = uuid.uuid4().hex
            responses.append(
                PredictionResponse(
                    prediction_id=pid,
                    churn_probability=round(p, 4),
                    churn_prediction=p >= threshold,
                    risk_level=_risk_level(p),
                    model_version=service.version,
                    latency_ms=per_item,
                )
            )
            rows.append(
                {
                    "id": pid,
                    "created_at": now,
                    "model_version": service.version,
                    "features": customer.model_dump(),
                    "probability": float(p),
                    "prediction": int(p >= threshold),
                    "latency_ms": per_item,
                }
            )
        app.state.store.log_many(rows)
        return responses, latency_ms

    # ------------------------------------------------------------------ endpoints
    @app.get("/", tags=["info"])
    def root() -> dict[str, Any]:
        return {
            "service": "churn-prediction-api",
            "model_version": app.state.model.version,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["info"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": app.state.model is not None,
            "model_version": app.state.model.version,
            "uptime_seconds": round(time.time() - state["started_at"], 1),
        }

    @app.get("/model/info", tags=["model"])
    def model_info() -> dict[str, Any]:
        service: ModelService = app.state.model
        return {**service.metadata, "loaded_at": service.loaded_at}

    @app.post("/model/reload", tags=["model"])
    def reload_model(x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        expected = os.getenv("ADMIN_TOKEN")
        if not expected or x_admin_token != expected:
            raise HTTPException(status_code=403, detail="invalid or missing admin token")
        app.state.model.load()
        return {"reloaded": True, "model_version": app.state.model.version}

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(customer: CustomerFeatures) -> PredictionResponse:
        responses, _ = _predict_and_log([customer])
        return responses[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(request: BatchRequest) -> BatchResponse:
        responses, latency = _predict_and_log(request.customers)
        return BatchResponse(
            model_version=app.state.model.version,
            count=len(responses),
            latency_ms=latency,
            predictions=responses,
        )

    @app.get("/monitoring/summary", tags=["monitoring"])
    def monitoring_summary(window_minutes: int = Query(60, ge=1, le=10_080)) -> dict[str, Any]:
        return {
            "model_version": app.state.model.version,
            **app.state.store.summary(window_minutes),
        }

    @app.get("/monitoring/drift", tags=["monitoring"])
    def monitoring_drift(window: int = Query(500, ge=50, le=20_000)) -> dict[str, Any]:
        current = app.state.store.recent_features(window)
        if len(current) < 50:
            return {
                "status": "insufficient_data",
                "detail": f"need at least 50 predictions, have {len(current)}",
                "current_rows": int(len(current)),
            }
        report = detect_drift(app.state.model.reference[FEATURES], current[FEATURES])
        report["status"] = "ok"
        report["model_version"] = app.state.model.version
        report["computed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        state["last_drift"] = report
        return report

    @app.get("/monitoring/recent", tags=["monitoring"])
    def monitoring_recent(limit: int = Query(50, ge=1, le=1000)) -> list[dict[str, Any]]:
        return app.state.store.recent(limit)

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def prometheus_metrics() -> str:
        """Prometheus-compatible metrics endpoint."""
        s = app.state.store.summary(5)
        drift = state["last_drift"]
        lines = [
            "# HELP predictions_total Total predictions served",
            "# TYPE predictions_total counter",
            f"predictions_total {s['total_predictions']}",
            "# HELP model_version Version of the model in production",
            "# TYPE model_version gauge",
            f"model_version {app.state.model.version}",
            "# HELP prediction_latency_ms_avg_5m Average latency over last 5 minutes",
            "# TYPE prediction_latency_ms_avg_5m gauge",
            f"prediction_latency_ms_avg_5m {s['avg_latency_ms'] or 0}",
            "# HELP prediction_positive_rate_5m Share of churn predictions over last 5 minutes",
            "# TYPE prediction_positive_rate_5m gauge",
            f"prediction_positive_rate_5m {s['positive_rate'] or 0}",
        ]
        if drift:
            lines += [
                "# HELP data_drift_share Share of drifted features (last computed)",
                "# TYPE data_drift_share gauge",
                f"data_drift_share {drift['share_drifted_features']}",
            ]
        return "\n".join(lines) + "\n"

    return app


app = create_app()
